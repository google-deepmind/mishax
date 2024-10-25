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

from collections.abc import Callable, Generator
import contextlib
import enum
import functools
import gc
import random
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from mishax import safe_greenlet


class SafeGreenletTest(parameterized.TestCase):

  def test_call_a_greenlet_that_calls_a_greenlet(self):
    def three():
      with safe_greenlet.SafeGreenlet(lambda: 3) as child:
        return child.switch()

    three_glet = safe_greenlet.SafeGreenlet(three)
    with self.assertRaisesRegex(RuntimeError, 'outside its context'):
      three_glet.switch()
    with three_glet:
      self.assertEqual(three_glet.switch(), 3)
    with self.assertRaisesRegex(RuntimeError, 'outside its context'):
      three_glet.switch()

  def test_non_reentrant(self):
    glet = safe_greenlet.SafeGreenlet(lambda: 1)

    def opener(glet):
      with glet:
        return glet.switch()

    with glet:
      with self.assertRaisesRegex(RuntimeError, 'already exists'):
        with glet:
          pass
      with safe_greenlet.SafeGreenlet(opener) as opener_glet:
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
          opener_glet.switch(glet)

  def test_self_switch_or_throw(self):
    self_switch = lambda *a, **kw: safe_greenlet.getcurrent().switch(*a, **kw)
    self_throw = lambda *a, **kw: safe_greenlet.getcurrent().throw(*a, **kw)
    with safe_greenlet.SafeGreenlet(self_switch) as glet:
      self.assertEqual(glet.switch(5), 5)
      self.assertTrue(glet.dead)
    with safe_greenlet.SafeGreenlet(self_switch) as glet:
      with self.assertRaisesRegex(ValueError, 'oops'):
        glet.throw(ValueError('oops'))
      self.assertTrue(glet.dead)
    with safe_greenlet.SafeGreenlet(self_throw) as glet:
      with self.assertRaisesRegex(ValueError, 'oops'):
        glet.switch(ValueError('oops'))
      self.assertTrue(glet.dead)
    with safe_greenlet.SafeGreenlet(self_throw) as glet:
      with self.assertRaisesRegex(ValueError, 'oops'):
        glet.throw(ValueError('oops'))
      self.assertTrue(glet.dead)

  def test_during_one_greenlet_call_a_function_that_uses_another(self):
    def two():
      with safe_greenlet.SafeGreenlet(lambda: 2) as child:
        return child.switch()

    def running_sum(total):
      while (x := safe_greenlet.getparent().switch(total)) is not None:
        total += x
      return total

    with safe_greenlet.SafeGreenlet(running_sum) as sum_glet:
      self.assertEqual(sum_glet.switch(1), 1)
      self.assertEqual(sum_glet.switch(two()), 3)
      self.assertEqual(sum_glet.switch(-1), 2)
      self.assertEqual(sum_glet.switch(None), 2)
      self.assertTrue(sum_glet.dead)

  def test_block_unreachable_child(self):
    def f():
      safe_greenlet.getparent().switch(1)
      safe_greenlet.getparent().switch(2)

    with safe_greenlet.SafeGreenlet(f) as f_glet:
      self.assertEqual(f_glet.switch(), 1)
      with safe_greenlet.SafeGreenlet(f_glet.switch) as call_glet:
        with self.assertRaisesRegex(RuntimeError, 'not reachable'):
          call_glet.switch()
        with self.assertRaisesRegex(RuntimeError, 'not reachable'):
          f_glet.switch()
      self.assertEqual(f_glet.switch(), 2)

  def test_custody_belongs_to_cleanup_greenlet(self):
    def open_and_wait(glet):
      with glet:
        safe_greenlet.getparent().switch()

    with safe_greenlet.SafeGreenlet(safe_greenlet.SafeGreenlet) as creator_glet:
      sub_glet = creator_glet.switch(lambda: safe_greenlet.getparent().switch())
    with self.assertRaisesRegex(RuntimeError, 'outside its context'):
      sub_glet.switch()
    with safe_greenlet.SafeGreenlet(open_and_wait) as opener_glet:
      opener_glet.switch(sub_glet)
      with safe_greenlet.SafeGreenlet(sub_glet.switch) as access_glet:
        with self.assertRaisesRegex(RuntimeError, 'not reachable'):
          access_glet.switch()
      sub_glet.switch()
      self.assertEqual(sub_glet.parent, opener_glet)

  def test_local_context(self):
    are_we_there_yet = False

    @contextlib.contextmanager
    def get_there():
      nonlocal are_we_there_yet
      are_we_there_yet = True
      try:
        yield
      finally:
        self.assertTrue(are_we_there_yet)
        are_we_there_yet = False

    def fn_using_local_context():
      try:
        with safe_greenlet.LocalContextManager(get_there):
          self.assertTrue(are_we_there_yet)
          try:
            return safe_greenlet.getparent().switch()
          finally:
            self.assertTrue(are_we_there_yet)
      finally:
        self.assertFalse(are_we_there_yet)

    with safe_greenlet.SafeGreenlet(fn_using_local_context) as glet:
      self.assertFalse(are_we_there_yet)
      glet.switch()
      self.assertFalse(are_we_there_yet)
      with self.assertRaises(StopIteration):
        glet.throw()
      self.assertFalse(are_we_there_yet)

  def test_clean_up_parricide(self):
    def child():
      with change_and_clean_up_state():
        safe_greenlet.getparent().switch()
        # On the second switch, kill grandparent.
        safe_greenlet.getparent().parent.throw()

    @contextlib.contextmanager
    def change_and_clean_up_state():
      current = safe_greenlet.getcurrent()
      # Change the shared state so we can check it's cleaned up by the time we
      # return.
      active_greenlets.append(current)
      # Register the greenlet's depth in the greenlet tree in `generations`.
      generations.append(sum(1 for _ in safe_greenlet._lineage(current)))
      try:
        yield
      finally:
        with self.subTest('no_child_left_at_end'):
          self.assertIsNone(safe_greenlet._REACHABLE_CHILD.get(None))
        with self.subTest('intervening_state_changes_cleaned_up'):
          self.assertEqual(active_greenlets.pop(), current)

    def switch_to_first_child_once_and_second_child_twice(child_fn=child):
      child1_glet = safe_greenlet.SafeGreenlet(child_fn)
      child2_glet = safe_greenlet.SafeGreenlet(child_fn)
      with change_and_clean_up_state(), child1_glet:
        child1_glet.switch()
        with child2_glet:
          child2_glet.switch()
          # Wait for second switch from parent.
          safe_greenlet.getparent().switch()
          child2_glet.switch()

    active_greenlets = []
    generations = []
    with safe_greenlet.SafeGreenlet(
        switch_to_first_child_once_and_second_child_twice
    ) as grandparent_glet:
      # grandparent_glet has 2 children, each trying to create two children and
      # switch to the first child once and the second one twice, but switch back
      # to grandparent after both children have been switched to once.
      grandparent_glet.switch(switch_to_first_child_once_and_second_child_twice)
      # At this point, grandparent_glet has switched to each of its children
      # once, and each child created its two children. Each greenlet registered
      # itself in `generations`, adding its depth in the greenlet tree: first
      # the grandparent (child of main greenlet), then its first child and two
      # grandchildren, then the second child and the other two grandchildren.
      self.assertListEqual(generations, [2, 3, 4, 4, 3, 4, 4])
      # Now, grandparent_glet will switch to its second child, which will switch
      # to its second child (the newest grandchild), which is running `child`;
      # therefore, it will throw a GreenletExit to the grandparent, which will
      # return `GreenletExit` to its parent, the main greenlet.
      retval = grandparent_glet.switch()
      self.assertIsInstance(retval, safe_greenlet.GreenletExit)
    with self.subTest('all_cleaned_up'):
      self.assertEmpty(active_greenlets)

  def test_no_interleaved_contexts_from_random_spawn_and_switch(self):
    active_greenlets = []
    root = safe_greenlet.getcurrent()

    @contextlib.contextmanager
    def register_active_greenlet():
      active_greenlets.append(safe_greenlet.getcurrent())
      try:
        yield
      finally:
        self.assertEqual(active_greenlets.pop(), safe_greenlet.getcurrent())

    def random_switch():
      reachables = list(safe_greenlet._reachable_progeny(root))
      self.assertIn(safe_greenlet.getcurrent(), reachables)
      # Allow jumping anywhere in the chain, but tend to stay near the end.
      # This distribution results in spawning 20-30 live greenlets in the maze
      # on average (not including ones that are never switched).
      target_idx = max(0, len(reachables) - 1 - int(random.expovariate(0.65)))
      reachables[target_idx].switch()

    def twisty_maze_of_passages_all_alike():
      with register_active_greenlet():
        with safe_greenlet.LocalContextManager(register_active_greenlet):
          with safe_greenlet.SafeGreenlet(twisty_maze_of_passages_all_alike):
            random_switch()
          with safe_greenlet.SafeGreenlet(twisty_maze_of_passages_all_alike):
            random_switch()
            with safe_greenlet.SafeGreenlet(twisty_maze_of_passages_all_alike):
              random_switch()
            random_switch()
          random_switch()

    for _ in range(1000):
      entrance = safe_greenlet.SafeGreenlet(twisty_maze_of_passages_all_alike)
      with entrance:
        entrance.switch()


class GeneratorMode(enum.Enum):
  # Run as an ordinary generator.
  PLAIN = enum.auto()
  # Run as an ordinary SafeGreenlet.
  GLET = enum.auto()
  # Run as a SafeGreenlet wrapped in `yield from`, using PEP 479 semantics.
  YIELD_FROM_GLET = enum.auto()
  # Yield from generator inside a SafeGreenlet.
  YIELD_FROM_AS_GLET = enum.auto()
  # Run with the `easy_greenlet` wrapper.
  EASY_GLET = enum.auto()

  def make_generator_fn(
      self,
      gen_fn: Callable[[], Generator[Any, Any, Any]],
      glet_run: Callable[[], Any],
  ) -> Callable[[], Generator[Any, Any, Any]]:
    """Returns a generator from the given generator function or greenlet run.

    These are assumed to have the same semantics, apart from the different
    conventions needed by generators (using `yield`) vs greenlets (using
    `safe_greenlet.yield_`).

    We use SafeGreenlet.__enter__() for some of these, to test that the GC can
    reliably close them, even though the intended usage pattern is to write a
    `with` block to make it clear when the greenlet will be closed.

    Args:
      gen_fn: A generator function.
      glet_run: An equivalent function using `safe_greenlet.yield_`.
    """
    if self == GeneratorMode.PLAIN:
      return gen_fn

    if self == GeneratorMode.YIELD_FROM_AS_GLET:
      run_yield_from = lambda: safe_greenlet.yield_from(gen_fn())
      return lambda: safe_greenlet.SafeGreenlet(run_yield_from).__enter__()

    if self == GeneratorMode.EASY_GLET:
      return lambda: safe_greenlet.easy_greenlet(glet_run).__enter__()

    glet_fn = lambda: safe_greenlet.SafeGreenlet(glet_run).__enter__()
    if self == GeneratorMode.GLET:
      return glet_fn

    if self == GeneratorMode.YIELD_FROM_GLET:

      def yield_from_glet():
        # Needed to capture and pass on the return value.
        return (yield from glet_fn())

      return yield_from_glet

    raise ValueError(f'Unknown generator mode: {self}')


class GeneratorTest(parameterized.TestCase):

  def subtest_with_generator(
      self,
      generator_mode: GeneratorMode,
      gen_fn: Callable[[], Generator[Any, Any, Any]],
      glet_run: Callable[[], Any],
      expect_successful_start: bool,
      subtest_body: Callable[[Generator[Any, Any, Any]], None],
  ) -> None:
    """Run a subtest with a generator, checking cleanups.

    Args:
      generator_mode: what kind of generator to run.
      gen_fn: a generator function that will be wrapped according to
        `generator_mode`.
      glet_run: a run function for a greenlet that will be wrapped according to
        `generator_mode`; should be semantically the same as `gen_fn`.
      expect_successful_start: whether the subtest is expected to successfully
        start the generator; we test that cleanup code is run iff this is True.
      subtest_body: a function testing the generator.
    """
    with self.subTest(subtest_body.__name__):
      def cleanup_gen_fn():
        nonlocal num_cleanups
        try:
          return (yield from gen_fn())
        finally:
          num_cleanups += 1

      def cleanup_glet_run():
        nonlocal num_cleanups
        try:
          return glet_run()
        finally:
          num_cleanups += 1

      num_cleanups = 0

      try:
        # Why do we call the subtest body here? So that there's no dangling
        # references to the generator from the test method, and we can check
        # that the generator is cleaned up properly.
        generator_fn = generator_mode.make_generator_fn(
            cleanup_gen_fn, cleanup_glet_run
        )
        subtest_body(generator_fn())
      finally:
        gc.collect()
        with self.subTest(f'{subtest_body.__name__}_cleanups'):
          self.assertEqual(num_cleanups, expect_successful_start)

  def set_up_helpers(self, generator_mode, gen_fn, glet_run):
    # This is for subtests where the generator is started and alive at some
    # point, which will result in the cleanup happening later on once the GC
    # finds the generator can be finalized and freed.
    self.subtest_with_start = functools.partial(
        self.subtest_with_generator, generator_mode, gen_fn, glet_run, True
    )
    # This is for subtests where the generator is closed or fails before
    # starting to run its code, resulting in the cleanup block never getting
    # hit.
    self.subtest_with_failing_start = functools.partial(
        self.subtest_with_generator, generator_mode, gen_fn, glet_run, False
    )

  @parameterized.parameters(GeneratorMode)
  def test_len_0_generators(self, generator_mode):
    def gen_fn():
      yield from ()
      return 2

    glet_run = lambda: 2
    self.set_up_helpers(generator_mode, gen_fn, glet_run)

    def iterate(generator):
      with self.assertRaises(StopIteration) as raised:
        next(generator)
      self.assertEqual(raised.exception.value, 2)
      with self.assertRaises(StopIteration) as raised:
        next(generator)
      self.assertIsNone(raised.exception.value)

    self.subtest_with_start(iterate)

    def send_nones(generator):
      with self.assertRaises(StopIteration) as raised:
        generator.send(None)
      self.assertEqual(raised.exception.value, 2)
      with self.assertRaises(StopIteration) as raised:
        generator.send(None)
      self.assertIsNone(raised.exception.value)

    self.subtest_with_start(send_nones)

    def send_not_none(generator):
      with self.assertRaises(TypeError):
        generator.send(5)

    self.subtest_with_failing_start(send_not_none)

    def throw(generator):
      with self.assertRaises(ValueError):
        generator.throw(ValueError())

    self.subtest_with_failing_start(throw)

    def close(generator):
      generator.close()
      with self.assertRaises(StopIteration) as raised:
        next(generator)
      self.assertIsNone(raised.exception.value)

    self.subtest_with_failing_start(close)

  @parameterized.parameters(GeneratorMode)
  def test_immediate_error_generators(self, generator_mode):
    def gen_fn():
      yield from ()
      raise ValueError()

    def glet_run():
      raise ValueError()

    self.set_up_helpers(generator_mode, gen_fn, glet_run)

    def iterate(generator):
      with self.assertRaises(ValueError):
        next(generator)
      with self.assertRaises(StopIteration) as raised:
        next(generator)
      self.assertIsNone(raised.exception.value)

    self.subtest_with_start(iterate)

    def throw(generator):
      with self.assertRaises(NameError):
        generator.throw(NameError())

    self.subtest_with_failing_start(throw)

    def close(generator):
      generator.close()
      with self.assertRaises(StopIteration) as raised:
        next(generator)
      self.assertIsNone(raised.exception.value)

    self.subtest_with_failing_start(close)

  @parameterized.parameters(GeneratorMode)
  def test_immediate_stopiteration_generators(self, generator_mode):
    def gen_fn():
      yield from ()
      raise StopIteration()

    def glet_run():
      raise StopIteration()

    self.set_up_helpers(generator_mode, gen_fn, glet_run)

    def iterate(generator):
      with self.assertRaises(RuntimeError):
        next(generator)

    self.subtest_with_start(iterate)

  @parameterized.parameters(GeneratorMode)
  def test_error_catching_generators(self, generator_mode):
    def gen_fn():
      try:
        yield 2
      except ValueError:
        pass
      yield (yield 3)

    def glet_run():
      try:
        safe_greenlet.yield_(2)
      except ValueError:
        pass
      safe_greenlet.yield_(safe_greenlet.yield_(3))

    self.set_up_helpers(generator_mode, gen_fn, glet_run)

    def iterate_throw_and_send(generator):
      self.assertEqual(next(generator), 2)
      self.assertEqual(generator.throw(ValueError()), 3)
      self.assertEqual(generator.send(4), 4)
      with self.assertRaises(StopIteration):
        next(generator)

    self.subtest_with_start(iterate_throw_and_send)


if __name__ == '__main__':
  absltest.main()
