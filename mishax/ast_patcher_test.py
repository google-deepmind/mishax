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

import ast
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


class RepatchDummy:

  def compute(self):
    x = 10
    x += 1
    return x


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


class FuzzyAnchorTarget:

  def call_with_drifted_kwarg(self, x, params):
    x = self._helper(
        x,
        params=params,
        scale=1.0,
        compute_in_f32=True,
    )
    return x

  def call_with_renamed_prefix(self, cache, new_cache, mask):
    cache = self._Utils.maybe_update(cache, new_cache, mask)
    return cache

  def call_with_3way_arg_merge(self, attn_vec, params):
    out = self._einsum(
        attn_vec,
        params,
        scale=2.0,
        extra_kwarg='upstream_added',
    )
    return out

  def return_with_added_element(self, x, cache, metrics):
    return x, cache, metrics, 'extra_upstream_item'

  def return_with_wrapped_tuple(self, out, new_cache):
    return (out, new_cache), 'params'

  def return_with_renamed_element(self, val, attn_cache, stats):
    return val, (attn_cache, stats)

  def if_with_drifted_else(self, x, flag):
    if flag:
      y = x + 1
    else:
      y = x - 1
      y = y * 2
    return y

  def ambiguous_calls(self, x, params):
    a = self._helper(x, params=params, mode='a', extra=1)
    a = self._helper(x, params=params, mode='b', extra=2)
    return a

  def call_with_multiple_splats(self, x, kw1, kw2):
    return self._helper(x, **kw1, **kw2)

  def nested_if_and_else_calls(self, x, mode):
    if mode == 'early':
      res = self._helper(x, tag='shared')
    elif mode == 'nested':
      if x > 0:
        res = self._helper(x, tag='shared')
      else:
        res = x
    else:
      res = self._helper(x, tag='else')
    return res

  def _helper(self, x, **kwargs):
    return x, kwargs

  class _Utils:

    @staticmethod
    def maybe_update(cache, new_cache, mask):
      return (cache, new_cache, mask)

  def _einsum(self, x, params, **kwargs):
    return (x, params, kwargs)


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
    with self.assertRaisesRegex(
        ast_patcher.PatchError, r'Too many \(2\) matches'
    ):
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

  def test_repatch_class_raises_patch_error(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE, RepatchDummy=['x += 1', 'x += 2']
    )
    with patcher():
      with self.assertRaisesRegex(
          ast_patcher.PatchError, r"does not define 'RepatchDummy'"
      ):
        ast_patcher.ModuleASTPatcher(MODULE, RepatchDummy=[])

  def test_ellipsis_call_and_rhs_rebind(self):
    # Statement patch with `...` matching remaining args and kwargs.
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'x = self._helper(x, ...)',
            'x = self._helper(x, ...)\nx = (x, "tagged")',
        ],
    )
    with patcher():
      res, tag = FuzzyAnchorTarget().call_with_drifted_kwarg(3, 'p')
      self.assertEqual(tag, 'tagged')
      self.assertEqual(
          res, (3, {'params': 'p', 'scale': 1.0, 'compute_in_f32': True})
      )

    # LHS rebinding (`_ = self._helper(x, ...)`) preserving captured kwargs.
    patcher_rhs = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'x = self._helper(x, ...)',
            '_ = self._helper(x, ...)\nx = (_, "rebound")',
        ],
    )
    with patcher_rhs():
      res, tag = FuzzyAnchorTarget().call_with_drifted_kwarg(5, 'p2')
      self.assertEqual(tag, 'rebound')
      self.assertEqual(
          res, (5, {'params': 'p2', 'scale': 1.0, 'compute_in_f32': True})
      )

  def test_ellipsis_call_arg_and_kwarg_modification(self):
    # Wrapping arg 0 and overriding a keyword argument while preserving `...`.
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'out = self._einsum(attn_vec, ..., scale=2.0)',
            'out = self._einsum((attn_vec, "wrapped"), ..., scale=4.0)',
        ],
    )
    with patcher():
      res = FuzzyAnchorTarget().call_with_3way_arg_merge('v', 'p')
      self.assertEqual(
          res,
          (
              ('v', 'wrapped'),
              'p',
              {'scale': 4.0, 'extra_kwarg': 'upstream_added'},
          ),
      )

    # Expression-level Call patch with `...`.
    patcher_expr = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'self._einsum(attn_vec, ...)',
            'self._einsum((attn_vec, "expr_wrapped"), ...)',
        ],
    )
    with patcher_expr():
      res = FuzzyAnchorTarget().call_with_3way_arg_merge('v', 'p')
      self.assertEqual(
          res,
          (
              ('v', 'expr_wrapped'),
              'p',
              {'scale': 2.0, 'extra_kwarg': 'upstream_added'},
          ),
      )

  def test_ellipsis_return_and_nested_tuples(self):
    patcher_added = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'return x, cache, ...',
            'metrics = (metrics, "flushed")\nreturn x, cache, ...',
        ],
    )
    with patcher_added():
      res = FuzzyAnchorTarget().return_with_added_element(1, 'c', 'm')
      self.assertEqual(res, (1, 'c', ('m', 'flushed'), 'extra_upstream_item'))

    patcher_wrapped = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'return (out, new_cache), ...',
            'new_cache = (new_cache, "wrapped")\nreturn (out, new_cache), ...',
        ],
    )
    with patcher_wrapped():
      res = FuzzyAnchorTarget().return_with_wrapped_tuple(1, 'nc')
      self.assertEqual(res, ((1, ('nc', 'wrapped')), 'params'))

    patcher_nested = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'return val, (..., stats)',
            'stats = (stats, "renamed")\nreturn val, (..., stats)',
        ],
    )
    with patcher_nested():
      res = FuzzyAnchorTarget().return_with_renamed_element(1, 'ac', 's')
      self.assertEqual(res, (1, ('ac', ('s', 'renamed'))))

  def test_ellipsis_if_statement(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'if flag:\n  ...\nelse:\n  ...',
            'if flag:\n  ...\nelse:\n  ...\ny += 100',
        ],
    )
    with patcher():
      self.assertEqual(FuzzyAnchorTarget().if_with_drifted_else(10, False), 118)
      self.assertEqual(FuzzyAnchorTarget().if_with_drifted_else(10, True), 111)

  def test_ellipsis_rejects_multiple_or_zero_matches(self):
    with self.assertRaisesRegex(
        ast_patcher.PatchError, r'Too many \(2\) matches'
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'a = self._helper(x, ...)',
              'a = self._helper(x, ...)\na += 1',
          ],
      )

    with self.assertRaisesRegex(ast_patcher.PatchError, 'No match in the AST'):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'x = self._helper(x, ..., nonexistent_kw=1)',
              'x = self._helper(x, ..., nonexistent_kw=1)',
          ],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError, 'Top-level `...` statements are not supported'
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'x = self._helper(x, ...)\n...',
              'x = self._helper(x, ...)\n...',
          ],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError, 'Top-level `...` statements are not supported'
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=['...', '...'],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError,
        'Replacement pattern contains `...` but match pattern does not',
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=['return y', 'return y, ...'],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError,
        'Replacement pattern has more `...` wildcards than match pattern',
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'x = self._helper(x, ...)',
              'x = self._helper(self._helper(x, ...), ...)',
          ],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError,
        'Match pattern has more `...` wildcards than replacement pattern',
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'a = ...\na = self._helper(x, ...)',
              'a = 0\na = self._helper(x, ...)',
          ],
      )

    with self.assertRaisesRegex(
        ast_patcher.PatchError,
        'Match pattern has more `...` wildcards than replacement pattern',
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'x = self._helper(x, ...)',
              'x = 0',
          ],
      )

  def test_ellipsis_override_implicitly_captured_kwarg(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'x = self._helper(x, ...)',
            'x = self._helper(x, ..., scale=9)',
        ],
    )
    with patcher():
      res, kw = FuzzyAnchorTarget().call_with_drifted_kwarg('val', {'w': 1})
      self.assertEqual(res, 'val')
      self.assertEqual(
          kw, {'params': {'w': 1}, 'scale': 9, 'compute_in_f32': True}
      )

  def test_ellipsis_multiple_splats_preserved(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'return self._helper(x, ...)',
            'return self._helper(x, ..., extra=42)',
        ],
    )
    with patcher():
      res, kw = FuzzyAnchorTarget().call_with_multiple_splats(
          'val', {'a': 1}, {'b': 2}
      )
      self.assertEqual(res, 'val')
      self.assertEqual(kw, {'a': 1, 'b': 2, 'extra': 42})

  def test_ellipsis_multiple_in_tuple_raises(self):
    with self.assertRaisesRegex(
        ast_patcher.PatchError, 'At most one `...` is allowed per sequence'
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'return x, cache, ...',
              'return x, ..., ...',
          ],
      )
    with self.assertRaisesRegex(
        ast_patcher.PatchError, 'At most one `...` is allowed per sequence'
    ):
      ast_patcher.ModuleASTPatcher(
          MODULE,
          FuzzyAnchorTarget=[
              'return x, ..., ...',
              'return x, ...',
          ],
      )
    stmt = ast.parse('(x, ..., ...)').body[0]
    assert isinstance(stmt, ast.Expr)
    with self.assertRaisesRegex(
        ast_patcher.PatchError, 'At most one `...` is allowed per sequence'
    ):
      ast_patcher._instantiate_ast_with_ellipsis(stmt.value, iter([]))

  def test_ellipsis_explicit_splat_deletion(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'return self._helper(x, ..., **kw1)',
            'return self._helper(x, ...)',
        ],
    )
    with patcher():
      res, kw = FuzzyAnchorTarget().call_with_multiple_splats(
          'val', {'a': 1}, {'b': 2}
      )
      self.assertEqual(res, 'val')
      self.assertEqual(kw, {'b': 2})

  def test_ellipsis_expr_in_call_and_tuple(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'out = ...',
            'out = self._helper(..., site="wrapped")',
        ],
    )
    with patcher():
      res = FuzzyAnchorTarget().call_with_3way_arg_merge('vec', 'params')
      self.assertEqual(
          res,
          (
              (
                  'vec',
                  'params',
                  {'scale': 2.0, 'extra_kwarg': 'upstream_added'},
              ),
              {'site': 'wrapped'},
          ),
      )

    patcher2 = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'out = ...',
            'out = (..., 42)',
        ],
    )
    with patcher2():
      res = FuzzyAnchorTarget().call_with_3way_arg_merge('vec', 'params')
      self.assertEqual(
          res,
          (
              (
                  'vec',
                  'params',
                  {'scale': 2.0, 'extra_kwarg': 'upstream_added'},
              ),
              42,
          ),
      )

  def test_ellipsis_call_with_nested_expr_and_call_wildcards(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            """
            a = ...
            a = self._helper(x, ...)
            """,
            """
            a = 0
            a = self._helper(tuple([..., 42]), ...)
            """,
        ],
    )
    with patcher():
      res = FuzzyAnchorTarget().ambiguous_calls('vec', 'params')
      self.assertEqual(
          res,
          (
              (
                  (
                      'vec',
                      {
                          'params': 'params',
                          'mode': 'a',
                          'extra': 1,
                      },
                  ),
                  42,
              ),
              {
                  'params': 'params',
                  'mode': 'b',
                  'extra': 2,
              },
          ),
      )

  def test_ellipsis_in_type_annotation_ignored(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        FuzzyAnchorTarget=[
            'x = self._helper(x, ...)',
            """
            x = self._helper(x, ...)
            extra: tuple[int, ...] = (1, 2)
            """,
        ],
    )
    with patcher():
      res, kw = FuzzyAnchorTarget().call_with_drifted_kwarg('val', {'w': 1})
      self.assertEqual(res, 'val')
      self.assertEqual(
          kw, {'params': {'w': 1}, 'scale': 1.0, 'compute_in_f32': True}
      )

  def test_ellipsis_multi_match_nested_if_else_order(self):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(
            allow_num_matches_upto={'FuzzyAnchorTarget': 3}
        ),
        FuzzyAnchorTarget=[
            'res = self._helper(x, tag=...)',
            'res = self._helper(x, tag=...)\nres = (res, "patched")',
        ],
    )
    with patcher():
      target = FuzzyAnchorTarget()
      self.assertEqual(
          target.nested_if_and_else_calls(1, 'early'),
          ((1, {'tag': 'shared'}), 'patched'),
      )
      self.assertEqual(
          target.nested_if_and_else_calls(1, 'nested'),
          ((1, {'tag': 'shared'}), 'patched'),
      )
      self.assertEqual(
          target.nested_if_and_else_calls(1, 'other'),
          ((1, {'tag': 'else'}), 'patched'),
      )


if __name__ == '__main__':
  absltest.main()
