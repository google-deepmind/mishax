# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An extension of Flax's nn.scan to support unstacked model internals access."""
import dataclasses
import functools
import inspect

from flax import linen as nn
from flax.core import scope
import jax
import jax.numpy as jnp


def sow(module: nn.Module, col, name, value, pred=None):
  if pred is None:
    reduce_fn = lambda prev, x: x
  else:
    reduce_fn = lambda prev, x: jax.lax.cond(pred, lambda: x, lambda: prev)
  init_fn = lambda: jnp.zeros_like(value)
  module.sow(col, name, value, reduce_fn=reduce_fn, init_fn=init_fn)


def instrumented_scan(
    *args,
    variable_zero_init_carry: scope.CollectionFilter,
    add_loop_index_col: str = 'loop',
    add_loop_index_name: str = 'index',
    **kwargs,
):
  """An nn.scan, with loop index var injected & selected vars 0-init carried."""
  bound = inspect.signature(nn.scan).bind(*args, **kwargs)
  bound.arguments['variable_carry'] = nn.module.union_filters(
      bound.arguments.get('variable_carry', False), variable_zero_init_carry
  )
  # assert not kwargs['check_constancy_invariants']
  # bound.arguments['check_constancy_invariants'] = False
  if add_loop_index_col:
    bound.arguments['variable_axes'] = {
        add_loop_index_col: 0,
        **bound.arguments['variable_axes'],
    }
  retval = nn.scan(*bound.args, **bound.kwargs)
  if add_loop_index_col:
    loop_indices = jnp.arange(bound.arguments['length'])
    maybe_loop_vars = {add_loop_index_col: {add_loop_index_name: loop_indices}}
    add_loop_idxs = lambda vars: vars | maybe_loop_vars
    retval = nn.map_variables(
        retval,
        add_loop_index_col,
        trans_in_fn=add_loop_idxs,
        trans_out_fn=lambda _: {},
    )
  else:
    maybe_loop_vars = {}
  orig_call = retval.__call__

  def call_with_zero_init_carry_vars(self, *args, **kwargs):
    if not self.is_initializing() and (
        collection_filter := scope.intersect_filters(
            variable_zero_init_carry, self.scope.mutable
        )
    ):
      shapes = jax.eval_shape(
          functools.partial(
              _vars_from_single_step,
              self,
              bound.arguments,
              collection_filter,
              maybe_loop_vars,
          ),
          *args,
          **kwargs,
      )
      for col, col_vars in shapes.items():
        for name, value in jax.tree.map(jnp.zeros_like, col_vars).items():
          self.put_variable(col, name, value)
    return orig_call(self, *args, **kwargs)

  retval.__call__ = call_with_zero_init_carry_vars
  return retval


def _take(x, i, axis):
  # Unlike jnp.take, this works even when x is an RNG key array.
  idx = [slice(None)] * x.ndim
  idx[axis] = i
  return x[*idx]


def _vars_from_single_step(
    scanned, scan_args, collection_filter, maybe_loop_vars, *args, **kwargs
):
  """Returns filtered vars from a single step of a `_scan`-ned module."""
  single_args = list(args)
  if len(args) - 1 != len(in_axes := scan_args.get('in_axes', ())):
    raise ValueError(
        f'Expected {len(args) - 1=} in_axes, got {len(in_axes)=}.'
    )
  for i, axis in enumerate(in_axes, 1):
    if axis != nn.broadcast:
      single_args[i] = jax.tree.map(
          lambda leaf, a=axis: _take(leaf, 0, a), args[i]
      )
  # Add loop index, if any.
  variables = dict(scanned.variables) | jax.tree.map(
      lambda x: x[0], maybe_loop_vars
  )
  # Slice scanned variables.
  for col in set(scanned.variables) & set(scan_args['variable_axes']):
    if (axis := scan_args['variable_axes'][col]) != nn.broadcast:
      variables[col] = jax.tree.map(
          lambda leaf, a=axis: _take(leaf, 0, a), scanned.variables[col]
      )
  _, variables = scan_args['target'](**{
      f.name: None if f.name == 'parent' else getattr(scanned, f.name)
      for f in dataclasses.fields(scanned)
  }).apply(variables, *single_args, mutable=collection_filter, **kwargs)
  return scope.group_collections(variables, [collection_filter])[0]
