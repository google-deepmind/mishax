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

"""AST patcher."""
import ast
import builtins
from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import importlib
import inspect
import os
import sys
import tempfile
import textwrap
import types
from typing import ContextManager
from unittest import mock
import weakref

import immutabledict


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PatchSettings:
  """Settings for ModuleASTPatcher.

  Attributes:
    prefix: A section of code to add near the top of the patched module, before
      the patched module members themselves.
    allow_num_matches_upto: The maximum number of places to apply each patch;
      specified as a mapping from member names. If this is not specified, or if
      no value is provided for a particular module member, the patcher will fail
      if there isn't exactly one match for each patch.
  """
  prefix: str | None = None
  allow_num_matches_upto: Mapping[str, int] | None = None


class PatchError(ValueError):
  """Raised when patches don't match the target source."""


def _ast_undump(dumped_ast: str) -> ast.AST:
  """Inverse of ast.dump."""
  return eval(dumped_ast, vars(ast) | vars(builtins))  # pylint: disable=eval-used


_INSTALLED_PATCHER_CONTEXTS = dict['ModuleASTPatcher', ContextManager[None]]()


class ModuleASTPatcher(Callable[[], ContextManager[None]]):
  """Creates a patcher that applies a series of patches to a module.

  The patched module members are in a separate temporary source file, visible to
  stacktraces and debuggers.

  `prefix` is code that should be run at the top of the file.

  For each object, the patches are logically a sequence of pairs (before, after)
  to match and replace; the patcher will error if there isn't at least one match
  (exactly one, unless more are specified via `settings`).
  For formatting convenience, the patcher will also accept a list of source
  blocks in alternating (before1, after1, before2, after2) order.

  The patcher may then be used as a `with patcher():` context manager, or using
  `patcher.install()` or `patcher.install_clean()`. These are in increasing
  order of robustness, and decreasing order of flexibility:
    * the `patcher()` context manager is flexible and useful for prototyping or
      debugging, but it changes and unchanges the target module, which isn't
      threadsafe.
    * `install_clean()` applies the patches on the target module as it's
      imported, which ensures consistency but requires the module to be
      specified as a string and not already imported.
    * `install()` is between the two. It's threadsafe, and can be called at any
      time by downstream code, but may occasionally produce incorrect results
      due to aliasing by other modules.

  NOTE: Patching a module on the source level may produce surprises additional
  to those implicitly mentioned above:
  1. If the target module is changed (e.g. via a version upgrade) then the patch
  could fail to apply, or (worse) apply but with changed meaning depending on
  the underlying code changes. Normally code maintainers will give consideration
  to downstream code by being careful with changes to the interface (e.g., type
  signatures), but will assume significant freedom in changing the
  implementation if the resulting behaviour is unchanged. However, this will
  still break patches, which depend on the implementation rather than the
  interface -- so even careful maintainers are under no obligation not to break
  patches.
  2. If the target module member definitions change global state (e.g. by
  registering themselves somewhere) then that state change will happen again
  when the patch is applied.
  """

  def __init__(
      self,
      module: types.ModuleType | str,
      settings: PatchSettings = PatchSettings(),
      /,
      **patches_per_object: str | Sequence[str] | Sequence[tuple[str, str]],
  ):
    self.module = module
    self._settings = settings
    self._patches_per_object = patches_per_object
    self._updated_members = None
    self._original_members = None
    self._src = None
    self._path = None
    self._globals = None
    if not isinstance(self.module, str):
      self._updated_members = self.updated_members

  def __str__(self):
    return f'ModuleASTPatcher({self.module_name})'

  def install(self) -> None:
    """Installs the patches on the target module.

    This is a cleaner way of using the patcher than via `__call__`, because it
    ensures code will subsequently always see the patched version of the module.

    It's less clean than `install_clean`; in exchange it allows the install to
    take place at some convenient time rather than necessarily having to precede
    the import of the patched module.

    However, transitioning between these different modes should be smooth -- you
    may .install_clean(), then later .install(), and enter a __call__() context,
    and the earlier things in this chain will turn the later ones into no-ops
    (if using the same ModuleASTPatcher instance).
    """
    if self not in _INSTALLED_PATCHER_CONTEXTS:
      context = self()
      context.__enter__()
      _INSTALLED_PATCHER_CONTEXTS[self] = context

  def is_installed(self) -> bool:
    """Returns whether the patcher is currently installed."""
    return self in _INSTALLED_PATCHER_CONTEXTS

  def install_clean(self) -> None:
    """Installs the patches on top of the freshly-imported module.

    This requires the module to be specified as a string, and will raise an
    error if the module is already imported. The benefit is that this ensures
    any other module that imports the patched module will get the patched
    version, rather than e.g. aliasing some unpatched module members, leading to
    subtle bugs.

    Usage:
      patcher = ModuleASTPatcher("some.fantastic.module", ...)
      patcher.install_clean()
    """
    if not isinstance(self.module, str) or self.module in sys.modules:
      raise ValueError(f'Module {self.module} already imported.')
    self.install()

  @contextlib.contextmanager
  def __call__(self):
    with contextlib.ExitStack() as stack:
      if self not in _INSTALLED_PATCHER_CONTEXTS:
        for name, value in self.updated_members.items():
          stack.enter_context(mock.patch.object(self.module, name, value))
      yield

  @property
  def module_name(self) -> str:
    return self.module if isinstance(self.module, str) else self.module.__name__

  @property
  def updated_members(self) -> immutabledict.immutabledict[str, object]:
    if self._updated_members is None:
      self._setup()
    self._updated_members: immutabledict.immutabledict[str, object]
    return self._updated_members

  @property
  def original_members(self) -> immutabledict.immutabledict[str, object]:
    if self._original_members is None:
      self._setup()
    self._original_members: immutabledict.immutabledict[str, object]
    return self._original_members

  @property
  def globals(self) -> dict[str, object]:
    if self._globals is None:
      self._setup()
    self._globals: dict[str, object]
    return self._globals

  @property
  def src(self) -> str:
    if self._src is None:
      self._setup()
    self._src: str
    return self._src

  @property
  def path(self) -> str:
    if self._path is None:
      self._setup()
    self._path: str
    return self._path

  def _setup(self):
    """Sets up the patcher source and temporary file."""
    if isinstance(self.module, str):
      module_name = self.module
      self.module = importlib.import_module(module_name)
    else:
      module_name = self.module.__name__
    self._original_members = immutabledict.immutabledict(
        {name: getattr(self.module, name) for name in self._patches_per_object}
    )
    src_blocks = [
        'import sys',
        (
            '# Import existing members of target module into patcher module.\n'
            f'globals().update(sys.modules[{module_name!r}].__dict__)'
        ),
    ]
    allow_num_matches_upto = self._settings.allow_num_matches_upto or {}
    if self._settings.prefix is not None:
      src_blocks.append(self._settings.prefix)
    for name in set(allow_num_matches_upto) - set(self._patches_per_object):
      raise ValueError(
          f'Unpatched module member {name} in settings.allow_num_matches_upto'
      )
    for name, patches in self._patches_per_object.items():
      target_src = inspect.getsource(getattr(self.module, name))
      dumped_ast = ast.dump(ast.parse(target_src))
      iter_patches = iter(patches)
      for patch in iter_patches:
        if isinstance(patch, str):
          # This is to support the convenient "flat" input format where multiple
          # patches can be provided as (before1, after1, before2, after2, ...).
          before, after = patch, next(iter_patches)
        else:
          before, after = patch
        before_src = textwrap.dedent(before).strip()
        after_src = textwrap.dedent(after).strip()
        before, after = ast.parse(before_src), ast.parse(after_src)

        match before.body:
          case [ast.Expr(before_expr_value)]:
            pass
          case _:
            before_expr_value = None
        match before_expr_value, after.body:
          case ast.expr(), [ast.Expr(after_expr_value)]:
            # Special case for patches that are expressions (rather than one or
            # more statements).
            before_dumped_ast = ast.dump(before_expr_value)
            after_dumped_ast = ast.dump(after_expr_value)
          case _:
            # In an AST dump with multiple statements, the statements will be
            # separated by ', '. An alternative is to ast.dump the before and
            # after nodes themselves, but that comes with more unwanted wrapping
            # text.
            before_dumped_ast = ', '.join(map(ast.dump, before.body))
            after_dumped_ast = ', '.join(map(ast.dump, after.body))
        num_matches = dumped_ast.count(before_dumped_ast)
        if not num_matches:
          if before_expr_value and ast.dump(before_expr_value) in dumped_ast:
            raise PatchError(
                f'The AST of {module_name}.{name} contains the AST of'
                f' ```\n{before_src}\n``` interpreted as an expression but not'
                ' as statement(s). This is an error because the replacement'
                ' code in the patch is statement(s), not an expression.'
            )
          else:
            raise PatchError(
                f'No match in the AST of {module_name}.{name} for the AST of'
                f' ```\n{before_src}\n```. The target source is:'
                f' ```\n{target_src}\n```.'
            )
        if num_matches > allow_num_matches_upto.get(name, 1):
          raise PatchError(
              f'Too many ({num_matches}) matches in the AST of'
              f' {module_name}.{name} for the AST of ```\n{before_src}\n```.'
              f' The target source is: ```\n{target_src}\n```. Consider'
              ' adjusting the PatchSettings.allow_num_matches_upto field.'
          )
        dumped_ast = dumped_ast.replace(before_dumped_ast, after_dumped_ast)
      patched_ast = ast.fix_missing_locations(_ast_undump(dumped_ast))
      src_blocks.append(ast.unparse(patched_ast))
    self._src = '\n\n'.join(src_blocks)
    # Create a file for stacktraces and debuggers to find source for patched
    # objects.
    fd, self._path = tempfile.mkstemp(
        prefix=f'{module_name}.', suffix='.PATCHED.py', text=True
    )
    weakref.finalize(self, os.remove, self._path)
    os.write(fd, self._src.encode())
    os.close(fd)
    exec(compile(self._src, self._path, 'exec'), envt := {})  # pylint: disable=exec-used
    self._globals = envt
    updated_members = {name: envt[name] for name in self._patches_per_object}
    self._updated_members = immutabledict.immutabledict(updated_members)
