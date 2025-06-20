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

"""An extension of JAX's lax.scan to support unstacked outputs."""
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard


@jt.jaxtyped(typechecker=typeguard.typechecked)
def scan_with_unstacked(
    f: Callable[..., Any],
    init: Any,
    xs: Any,
    length: int,
    unstack: jt.PyTree[bool] = True,
    **kwargs,
) -> Tuple[Any, Any]:
  """A jax.lax.scan, with optional unstacking of outputs.

  This function avoids materializing the stacked outputs for which `unstack`
  is True. It does so by accumulating the unstacked outputs in the carry.

  Args:
    f: The scan body function.
    init: The initial carry state.
    xs: The input sequence.
    length: The length of the sequence. Mandatory.
    unstack: A JAX tree prefix of the scan body's output, with bool leaves. For
      each True, the corresponding output will be a tuple of length `length`
      with the unstacked outputs; while for each False, the output will be
      unchanged (stacked). If `unstack` is the boolean True, the output will be
      transposed into a tuple of output pytrees.
    **kwargs: Additional keyword arguments for jax.lax.scan.

  Returns:
    A tuple (carry, ys) where `carry` is the final carry state and `ys` is
    the output sequence, potentially unstacked according to `unstack`.
  """
  unstack_leaves, unstack_struct = jax.tree.flatten(unstack)

  def body(carry_and_accumulators, x_and_index):
    carry, accumulators = carry_and_accumulators
    x, index = x_and_index
    carry, y = f(carry, x)
    ys_to_unstack = []
    ys_to_stack = []
    for unstack, part in zip(unstack_leaves, unstack_struct.flatten_up_to(y)):
      (ys_to_unstack if unstack else ys_to_stack).append(part)
    if accumulators is None:  # This is to produce the right eval_shape result.
      accumulators = [ys_to_unstack] * length
    else:
      preds = jnp.arange(length) == index
      accumulators = [
          jax.lax.cond(pred, lambda: ys_to_unstack, lambda a=acc: a)
          for pred, acc in zip(preds, accumulators)
      ]
    return (carry, accumulators), ys_to_stack

  x_shape = jax.eval_shape(lambda xs: jax.tree.map(lambda x: x[0], xs), xs)
  (_, unstacked_ys_shape), _ = jax.eval_shape(body, (init, None), (x_shape, 0))
  unstacked_ys = jax.tree.map(jnp.zeros_like, unstacked_ys_shape)
  (carry, unstacked_ys), stacked_ys = jax.lax.scan(
      body, (init, unstacked_ys), (xs, jnp.arange(length)), length, **kwargs
  )
  unstacked_ys = iter(zip(*unstacked_ys))
  stacked_ys = iter(stacked_ys)
  ys = unstack_struct.unflatten([
      next(unstacked_ys if unstack else stacked_ys)
      for unstack in unstack_leaves
  ])
  return carry, ys
