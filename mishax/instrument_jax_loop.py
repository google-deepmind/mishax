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
from typing import Any, Callable

import jax
import jax.numpy as jnp


def scan_with_unstacked(
    f: Callable[..., Any],
    init: Any,
    xs: Any,
    length: int | None = None,
    unstack: Any = True,
    **kwargs,
) -> tuple[Any, Any]:
  """A jax.lax.scan, with optional unstacking of outputs.

  This function avoids materializing the stacked outputs for which `unstack`
  is True. It does so by accumulating the unstacked outputs in the carry. This
  reduces memory usage via dead code elimination (DCE) if the unstacked outputs
  are sometimes selectively discarded.

  Args:
    f: The scan body function.
    init: The initial carry state.
    xs: The input sequence.
    length: The length of the sequence.
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
  if length is None:
    x_lengths = {len(x) for x in jax.tree.leaves(xs)}
    if len(x_lengths) != 1:
      raise ValueError(f'Could not infer length from xs: found {x_lengths=}.')
    [length] = x_lengths

  unstack_leaves, unstack_treedef = jax.tree.flatten(unstack)

  def body(carry_and_accumulators, x_and_index):
    carry, accumulators = carry_and_accumulators
    x, index = x_and_index

    carry, y = f(carry, x)

    ys_to_unstack = []
    ys_to_stack = []
    for unstack, part in zip(unstack_leaves, unstack_treedef.flatten_up_to(y)):
      (ys_to_unstack if unstack else ys_to_stack).append(part)

    if accumulators is None:  # This is to produce the right eval_shape result.
      accumulators = [ys_to_unstack] * length
    else:
      preds = jnp.arange(length) == index
      # This may look like it's O(length^2) (across the full scan) because it's
      # O(length) in every step, but actually that's mostly false: there's only
      # O(length) array updates in total; the potential O(length^2) factor comes
      # from the conditional checks, which are each cheap.
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

  unstacked_ys_iter = iter(zip(*unstacked_ys))
  stacked_ys_iter = iter(stacked_ys)
  ys = unstack_treedef.unflatten([
      next(unstacked_ys_iter if unstack else stacked_ys_iter)
      for unstack in unstack_leaves
  ])
  return carry, ys
