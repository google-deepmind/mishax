# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library for nonlocal `yield`, and jittable concurrency.

This is a wrapper for the [greenlet](https://greenlet.rtfd.io) library, which
offers "lightweight coroutines for in-process sequential concurrent programming"
-- called "greenlets". We expose a restricted subset of greenlet functionality,
to make it easier to write correct code, in particular in the context of Jax
code that needs to remain jittable (e.g. avoid side effects).

Greenlets are a powerful concurrency primitive. However, they may generally be
too powerful, and make it hard to reason about code behaviour -- they're similar
to gotos in this way.

For example: sometimes a function needs to modify global state, and then clean
it up. If using gotos for control flow, a program might start a "function" f,
then later start a "function" g that modifies the same global state, then
accidentally return to the f cleanup code before g has had a chance to clean up.
Then f's cleanup code is likely to fail.

Ordinarily, structured programming ensures that function calls don't
overlap/interleave in this way: the call to g would occur either inside f (and
completing before f makes further progress), or before or after the call to f.

Coroutine-based programming can violate this non-interleaving assumption, e.g.,
with a generator as follows:

```
def f():
  try:
    make_a_global_change()
    ...
    yield some value
    ...
  finally:
    clean_up_the_change()

# Inter-generator interference:
for v1, v2 in zip(f(), f()):
  pass

# Caller-generator interference:
try:
  make_a_global_change()
  for v in f():
    ...
    if something_true():
      break
finally:
  # Failure here.
  clean_up_the_change()
```

The ways to avoid these and adhere to the non-interleaving assumption:
1. Ensure each generator must clean up global state before yielding.
2. If a generator must change global state, ensure global state isn't modified
  between calls to the generator. This means e.g., the inter-generator
  interference example must produce an error; in addition the caller must make
  sure to avoid creating caller-generator interference, by cleaning up any
  global state changes before re-transferring control to the generator.

Our [motivating use
case](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Flexibility__using_greenlets_)
for coroutines is to instrument code locations in an LLM forward pass; here's a
sketch of how this would work:

```
def instrumented_forward_pass(inputs, params):
  ...
  # Encountered an Activation site we'd like to expose!
  yield site, mutable_value
  # Resume, with the modified value ...
```

However, there are two complications:
* the instrumentation site is deep in a call stack, so `yield` isn't available;
to enable running in a coroutine without having to change the whole call stack
(as with asyncio or generators) or lose the global state (as with threads), we
can use `safe_greenlet.yield_`.
* the call stack involves complex global state, both for Jax (e.g. tracer state)
and for the framework (e.g. Haiku or Flax); so solution 1 is off the table, but
we can facilitate solution 2 using safe_greenlet.

The Python language feature that best typifies how to ensure cleanup code is run
in the correct order is the `with` block, which uses a context manager; if code
code is inside multiple `with` blocks, the inner one is always cleaned up before
continuing with the outer.

We will state 3 ground rules, and enforce the first 2:

* Each greenlet must have at most one child greenlet it can switch/throw to at
any one time (we'll call it the *reachable child*), as determined by a
non-reentrant context manager to prevent interleaving.
* The context manager will also ensure the reachable child exits.
* Finally, the calling code must ensure global state is kept consistent between
subsequent switches/throws to the same greenlet.

To make it easier to keep state consistent but different in different greenlets,
we have a `LocalContextManager` context manager wrapper class, that exits and
reenters a context whenever execution leaves and reenters its greenlet.

Since `yield_` is sufficient for our existing greenlet use case, we've
made SafeGreenlets fully conform to Generator behaviour, and extend Generator.

Finally, since the PEP 380 way of exposing return values via `StopIteration` and
the PEP 342 way of sending/injecting values using a separate send method are not
very ergonomic, we provide an `EasyGenerator` wrapper with .sendval and .retval
accessors, and an `easy_greenlet` entry point to create SafeGreenlets with these
conveniences as well.
"""

from collections.abc import Callable, Generator, Iterable, Iterator
import contextlib
import contextvars
import functools
import typing
from typing import Any, ContextManager, TypeVar
import weakref

import greenlet


_REACHABLE_CHILD = contextvars.ContextVar[
    weakref.ReferenceType['SafeGreenlet']
]('reachable_child')
_LOCAL_CONTEXTS = contextvars.ContextVar[list['LocalContextManager']](
    'local_contexts'
)

getcurrent = greenlet.getcurrent
getparent = lambda: getcurrent().parent

# The default exception thrown by `greenlet.throw`; causes the receiving
# greenlet to return GreenletExit.
GreenletExit = greenlet.GreenletExit

_YieldT = TypeVar('_YieldT')
_SendT = TypeVar('_SendT')
_ReturnT = TypeVar('_ReturnT')


class _Sentinel:
  pass


_NOT_PROVIDED = _Sentinel()


def yield_(*args, default: Any = _NOT_PROVIDED) -> Any:
  if parent := getparent():
    # Why switch rather than send? 2 reasons.
    # 1. parent may not be a SafeGreenlet, eg the main greenlet isn't.
    # 2. The semantics only differs if parent isn't started (impossible) or if
    #    parent is finishing (which means this greenlet will be killed rather
    #    than receiving another value).
    return parent.switch(*args if args else [None])
  # We're in the main greenlet, with no parent.
  if default is _NOT_PROVIDED:
    raise RuntimeError('`yield_` called without default from main greenlet!')
  return default


def easy_greenlet(
    run: Callable[..., _ReturnT] | None = None,
) -> ContextManager['EasyGenerator[Any, Any, _ReturnT]']:
  """Run `run` in a SafeGreenlet, wrapped in an EasyGenerator for convenience."""

  @contextlib.contextmanager
  def wrapped():
    # `ctx` passed so the greenlet owns a reference to the context, keeping it
    # from getting GC-d prematurely.
    with SafeGreenlet(lambda ctx=ctx: run()) as glet:
      yield EasyGenerator(glet)

  ctx = wrapped()
  return ctx


class SafeGreenlet(greenlet.greenlet, Generator[_YieldT, _SendT, _ReturnT]):
  """Context-managed greenlet that enforces the one-reachable-child constraint.

  The constraint is that if we're using `SafeGreenlet`s everywhere, a greenlet
  can only be switched or thrown to either:
  1. In the greenlet's own context (with no other SafeGreenlet interleaving),
     from an ancestor.
  2. If the greenlet is an ancestor of the switching/throwing greenlet. When
     this happens, all local contexts (created using `LocalContextManager`)
     downstream of the greenlet are temporarily closed, until such time as the
     downstream greenlet(s) are reentered.

  `greenlet` allows a greenlet's parent to be set freely, but by default sets it
  to the greenlet that creates the child greenlet. In contrast, the parent of a
  `SafeGreenlet` is the greenlet that opens its context.
  """

  def __init__(self, run: Callable[..., _ReturnT] | None = None):
    self._context_token: (
        contextvars.Token[weakref.ReferenceType['SafeGreenlet']] | None
    ) = None
    super().__init__(run)

  def __enter__(self) -> 'SafeGreenlet[_YieldT, _SendT, _ReturnT]':
    if self._context_token is not None:
      raise RuntimeError('Context already exists!')
    self._context_token = _REACHABLE_CHILD.set(weakref.ref(self))
    super().__setattr__('parent', getcurrent())
    return self

  def __exit__(self, *args):
    try:
      if self:
        self.close()
    finally:
      _REACHABLE_CHILD.reset(self._context_token)
      self._context_token = None

  def throw(self, *args, **kwargs) -> _YieldT:
    return _managed_switch(self, super().throw, *args, **kwargs)

  def _switch(self, *args, **kwargs) -> _YieldT:
    return _managed_switch(self, super().switch, *args, **kwargs)

  def switch(self, *args, **kwargs) -> _YieldT | _ReturnT:
    try:
      return self._switch(*args, **kwargs)
    except StopIteration as e:
      return e.value

  def __setattr__(self, name, value):
    if name == 'parent':
      raise RuntimeError('Changing parent is not permitted!')
    super().__setattr__(name, value)

  def __bool__(self) -> bool:  # pylint: disable=useless-parent-delegation
    """Returns True if active and False if dead or not yet started."""
    return super().__bool__()

  @property
  def dead(self) -> bool:
    """True if this greenlet is dead (i.e., it finished its execution)."""
    return super().dead

  # Generator methods, as per PEP 342

  def send(self, value: _SendT) -> _YieldT:
    if self.dead:
      raise StopIteration()
    if self:
      return self._switch(value)
    if value is not None:
      raise TypeError("Can't send non-None value to a just-started generator.")
    return self._switch()

  def close(self) -> None:
    try:
      super().close()
    finally:
      if self:
        raise RuntimeError('generator ignored GeneratorExit')


class EasyGenerator(Generator[_YieldT, _SendT, _ReturnT]):
  """Returns a wrapper providing an easier interface to a generator.

  The wrapper has the same behaviour as `inner_gen`, but with an alternative way
  to send values and get the final returned value, by specifying `gen.sendval`
  and reading `gen.retval`, which can work with a regular loop.
  """
  sendval: _SendT = None
  retval: _ReturnT | None = None

  def __init__(self, inner_gen: Generator[_YieldT, _SendT, _ReturnT]):
    self.inner_gen = inner_gen

  @contextlib.contextmanager
  def _reset_sendval_and_capture_retval(self) -> Iterator[None]:
    self.sendval = None
    try:
      yield
    except StopIteration as e:
      self.retval = e.value
      raise e

  def send(self, value: _SendT) -> _YieldT:
    with self._reset_sendval_and_capture_retval():
      return self.inner_gen.send(value)

  def __next__(self) -> _YieldT:
    return self.send(self.sendval)

  def throw(self, *args, **kwargs) -> _YieldT:
    with self._reset_sendval_and_capture_retval():
      return self.inner_gen.throw(*args, **kwargs)

  def close(self) -> None:
    return self.inner_gen.close()


class LocalContextManager(ContextManager[Any]):
  """Context manager keeping a context open only in the current greenlet."""

  def __init__(self, context_manager_fn: Callable[[], ContextManager[Any]]):
    """Constructor.

    Args:
      context_manager_fn: A function that returns a context manager. This will
        be called to get a fresh context manager whenever control returns to the
        greenlet.
    """
    self._context_manager_fn = context_manager_fn
    self._context_manager = None
    self._glet = None

  def __enter__(self) -> Any:
    if self._glet is not None:
      raise RuntimeError('Local context already entered!')
    if _LOCAL_CONTEXTS.get(None) is None:
      _LOCAL_CONTEXTS.set([])
    _LOCAL_CONTEXTS.get().append(self)
    self._glet = getcurrent()
    return self._open()

  def _open(self) -> Any:
    if getcurrent() != self._glet:
      raise RuntimeError('Local context opened from wrong greenlet!')
    if self._context_manager is not None:
      raise RuntimeError('Context already open!')
    self._context_manager = self._context_manager_fn()
    return self._context_manager.__enter__()

  def _close(self, exc_typ=None, exc_val=None, exc_tb=None):
    if getcurrent() != self._glet:
      raise RuntimeError('Local context closed from wrong greenlet!')
    if self._context_manager is None:
      raise RuntimeError('Context already closed!')
    try:
      self._context_manager.__exit__(exc_typ, exc_val, exc_tb)
    finally:
      self._context_manager = None

  @contextlib.contextmanager
  def close_temporarily(self):
    """Closes the context, and reopens a fresh one on exit.

    This is for use in `_managed_switch`, where we tidy away the local contexts
    of the calling greenlet before switching to a different one, and then
    restore them on return.

    Yields:
      None, after the local context has been closed.
    """
    self._close()
    try:
      yield
    finally:
      self._open()

  def __exit__(self, exc_typ=None, exc_val=None, exc_tb=None):
    try:
      self._close(exc_typ, exc_val, exc_tb)
    finally:
      if getcurrent() != self._glet:
        raise RuntimeError('Local context exiting from wrong greenlet!')
      if _LOCAL_CONTEXTS.get()[-1] != self:
        raise RuntimeError('Local context closed in wrong order!')
      _LOCAL_CONTEXTS.get().pop()
      self._glet = None


@typing.overload
def yield_from(
    iterable: Iterable[Any],
    default_value_to_send: None | _Sentinel = _NOT_PROVIDED,
) -> None:
  ...


def yield_from(
    iterable: Generator[Any, _SendT, _ReturnT],
    default_value_to_send: _SendT | _Sentinel = _NOT_PROVIDED,
) -> _ReturnT:
  """An equivalent to builtin `yield from` that can be called in a greenlet."""
  # The implementation is based on the `yield from` definition in PEP 380.
  iterator = iter(iterable)
  try:
    value_to_yield = next(iterator)
  except StopIteration as e:
    return e.value
  else:
    while True:
      try:
        value_to_send = yield_(value_to_yield, default=default_value_to_send)
      except GeneratorExit as e:
        try:
          close_method = iterator.close  # pytype: disable=attribute-error
        except AttributeError:
          pass
        else:
          close_method()
        raise e
      except BaseException as e:  # pylint: disable=broad-exception-caught
        try:
          throw_method = iterator.throw  # pytype: disable=attribute-error
        except AttributeError:
          raise e  # pylint: disable=raise-missing-from
        else:
          try:
            value_to_yield = throw_method(e)
          except StopIteration as stop_it:
            return stop_it.value
      else:
        try:
          if value_to_send is None:
            value_to_yield = next(iterator)
          else:
            value_to_yield = iterator.send(value_to_send)  # pytype: disable=attribute-error
        except StopIteration as stop_it:
          return stop_it.value


def _reachable_progeny(glet: greenlet.greenlet) -> Iterator[SafeGreenlet]:
  while (
      (context := glet.gr_context)
      and (glet_ref := context.get(_REACHABLE_CHILD)) is not None
      and (glet := glet_ref()) is not None
  ):
    yield glet


def _lineage(glet: greenlet.greenlet) -> Iterator[greenlet.greenlet]:
  yield glet
  while glet := glet.parent:
    yield glet


def _managed_switch(
    glet: greenlet.greenlet, switch_or_throw_method, *args, **kwargs
) -> Any:
  """Runs the given method call, with safety checks and local context switching."""
  # Ensure SafeGreenlet context is open.
  if isinstance(glet, SafeGreenlet) and glet._context_token is None:  # pylint: disable=protected-access
    raise RuntimeError('Target greenlet is outside its context!')
  current = getcurrent()
  # If no SafeGreenlets involved, fall through to normal greenlet behaviour.
  if not (isinstance(glet, SafeGreenlet) or isinstance(current, SafeGreenlet)):
    return switch_or_throw_method(*args, **kwargs)
  # Enforce reachability constraint.
  if not (glet in _reachable_progeny(current) or glet in _lineage(current)):
    raise RuntimeError('Target greenlet is not reachable!')
  with _close_and_reopen_local_context_managers():
    try:
      retval = switch_or_throw_method(*args, **kwargs)
    except StopIteration as e:
      raise RuntimeError(
          'Raising StopIteration in a generator is not permitted, see PEP 749.'
      ) from e
    else:
      if not glet:
        raise StopIteration(retval)
      return retval


def _close_and_reopen_local_context_managers():
  if ctxs := _LOCAL_CONTEXTS.get(None):
    retval = contextlib.ExitStack()
    for ctx in ctxs[::-1]:
      retval.enter_context(ctx.close_temporarily())
    return retval
  return contextlib.nullcontext()


_main_glet = getcurrent()
_main_glet.switch = functools.partial(
    _managed_switch, _main_glet, greenlet.greenlet.switch, _main_glet
)
_main_glet.throw = functools.partial(
    _managed_switch, _main_glet, greenlet.greenlet.throw, _main_glet
)
