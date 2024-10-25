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

import enum
import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.tree_util

from mishax import ast_patcher


class PlainClass:
  def __init__(self):
    x = 0
    x += 2
    self.x = x

  def sum_of_two_xs(self):
    return self.x + self.x

  @classmethod
  def get_one(cls):
    return cls()


# Using decorators and an implicit metaclass.
@jax.tree_util.register_static
class FancyClass(enum.Enum):
  A = enum.auto()

  @property
  def x(self):
    x = 0
    x += 2
    return x

  def sum_of_two_xs(self):
    return self.x + self.x

  @classmethod
  def get_one(cls):
    return cls.A


def hit_patch():
  global HIT_PATCH
  HIT_PATCH = True


HIT_PATCH = False
UndecoratedEnum = enum.Enum('UndecoratedEnum', ['A'])
MODULE = sys.modules[__name__]


class AstPatcherTest(parameterized.TestCase):

  @parameterized.product(
      cls_name=[PlainClass.__name__, FancyClass.__name__], install=[True, False]
  )
  def test_patch(self, cls_name: str, install: bool):
    self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
    global HIT_PATCH
    HIT_PATCH = False
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        **{cls_name: ['x += 2', 'x += 2\nhit_patch()']},
    )
    if install:
      self.assertFalse(patcher.is_installed())
      patcher.install()
      self.assertTrue(patcher.is_installed())
    for i in range(2):
      with self.subTest(['first_time', 'reuse'][i]), patcher():
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertTrue(HIT_PATCH)
        HIT_PATCH = False
      with self.subTest('after ' + ['second_time', 'reuse'][i]):
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertEqual(HIT_PATCH, install)
    if install:
      self.assertTrue(patcher.is_installed())
      del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]
      self.assertFalse(patcher.is_installed())

  def test_decorator_still_applied(self):
    orig_fancy_class = FancyClass
    patcher = ast_patcher.ModuleASTPatcher(MODULE, FancyClass=[])
    with patcher():
      self.assertNotEqual(orig_fancy_class, FancyClass)
      jax.jit(lambda x: x)(FancyClass.A)
    with self.subTest('decorator_was_needed'), self.assertRaises(TypeError):
      jax.jit(lambda x: x)(UndecoratedEnum.A)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_patch_expr_and_prefix(self, cls_name):
    self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='FOUR = 4'),
        **{cls_name: ['2', 'FOUR']},
    )
    for i in range(2):
      with self.subTest(['first_time', 'reuse'][i]), patcher():
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 4)
      with self.subTest('after ' + ['second_time', 'reuse'][i]):
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_stacktrace(self, cls_name):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='import inspect'),
        **{cls_name: ['x', 'inspect.getsource(inspect.currentframe())']},
    )
    with self.subTest('src_match'), patcher():
      self.assertContainsExactSubsequence(
          patcher.src, getattr(MODULE, cls_name).get_one().x
      )
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='import inspect'),
        **{cls_name: ['x', 'inspect.getsourcefile(inspect.currentframe())']},
    )
    with self.subTest('src_path'), patcher():
      self.assertEqual(patcher.path, getattr(MODULE, cls_name).get_one().x)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_no_double_patch_by_default(self, cls_name):
    with self.assertRaisesRegex(ValueError, 'Too many matches'):
      patcher = ast_patcher.ModuleASTPatcher(
          MODULE, **{cls_name: ['self.x', 'self.x + 1']}
      )
      with patcher():
        pass

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_can_double_patch(self, cls_name):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(allow_num_matches_upto={cls_name: 2}),
        **{cls_name: ['self.x', 'self.x + 1']},
    )
    with patcher():
      self.assertEqual(getattr(MODULE, cls_name).get_one().sum_of_two_xs(), 6)

  def test_install_clean(self):
    patcher = ast_patcher.ModuleASTPatcher(
        'mishax.safe_greenlet',
        ast_patcher.PatchSettings(allow_num_matches_upto=dict(yield_=2)),
        yield_=['default', '"peekaboo"'],
    )
    patcher.install_clean()

    from mishax import safe_greenlet  # pylint: disable=g-import-not-at-top

    transient_patcher = ast_patcher.ModuleASTPatcher(
        'mishax.safe_greenlet',
        ast_patcher.PatchSettings(allow_num_matches_upto=dict(yield_=2)),
        yield_=['"peekaboo"', '"THUMP"'],
    )

    with self.subTest('plain'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    with self.subTest('with_context'), transient_patcher():
      self.assertEqual(safe_greenlet.yield_(), 'THUMP')

    with self.subTest('after_context'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    patcher.install()
    with self.subTest('after_install'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    with self.subTest('after_install_with_context'), transient_patcher():
      self.assertEqual(safe_greenlet.yield_(), 'THUMP')

    with self.subTest('after_install_and_context'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

if __name__ == '__main__':
  absltest.main()
