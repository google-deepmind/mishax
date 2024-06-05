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

"""Tests for ast_patcher."""

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

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_patch(self, cls_name):
    self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
    global HIT_PATCH
    self.assertFalse(HIT_PATCH)
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        **{cls_name: ['x += 2', 'x += 2\nhit_patch()']},
    )
    for i in range(2):
      with self.subTest(['first_time', 'reuse'][i]), patcher():
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertTrue(HIT_PATCH)
        HIT_PATCH = False
      with self.subTest('after ' + ['second_time', 'reuse'][i]):
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertFalse(HIT_PATCH)

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


if __name__ == '__main__':
  absltest.main()
