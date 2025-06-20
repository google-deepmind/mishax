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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from mishax import instrument_jax_loop
import numpy as np


class InstrumentJaxLoopTest(absltest.TestCase):

  def test_scan_with_unstacked(self):
    def body_fn(carry, x):
      return carry + x, (carry * 2, x * 3)

    init_carry = 0
    xs = jnp.arange(5)
    length = 5

    # Test with unstack=True (unstack all)
    carry_all_unstacked, ys_all_unstacked = (
        instrument_jax_loop.scan_with_unstacked(
            body_fn, init_carry, xs, length=length, unstack=True
        )
    )

    self.assertEqual(carry_all_unstacked, sum(xs))
    self.assertLen(ys_all_unstacked, length)
    self.assertLen(ys_all_unstacked[0], 2)
    self.assertLen(ys_all_unstacked[1], 2)
    self.assertLen(ys_all_unstacked[2], 2)
    self.assertLen(ys_all_unstacked[3], 2)
    self.assertLen(ys_all_unstacked[4], 2)

    # Verify unstacked outputs
    expected_ys0 = []
    expected_ys1 = []
    current_carry = init_carry
    for x_val in xs:
      current_carry, (y0, y1) = body_fn(current_carry, x_val)
      expected_ys0.append(y0)
      expected_ys1.append(y1)

    for i in range(length):
      np.testing.assert_array_equal(ys_all_unstacked[i][0], expected_ys0[i])
      np.testing.assert_array_equal(ys_all_unstacked[i][1], expected_ys1[i])

    # Test with unstack=False (keep all stacked)
    carry_all_stacked, ys_all_stacked = instrument_jax_loop.scan_with_unstacked(
        body_fn, init_carry, xs, length=length, unstack=False
    )

    self.assertEqual(carry_all_stacked, sum(xs))
    self.assertLen(ys_all_stacked, 2)
    np.testing.assert_array_equal(ys_all_stacked[0], jnp.array(expected_ys0))
    np.testing.assert_array_equal(ys_all_stacked[1], jnp.array(expected_ys1))

    # Test with mixed unstacking (True for first output, False for second)
    unstack_mixed = (True, False)
    carry_mixed, ys_mixed = instrument_jax_loop.scan_with_unstacked(
        body_fn, init_carry, xs, length=length, unstack=unstack_mixed
    )

    self.assertEqual(carry_mixed, sum(xs))
    self.assertLen(ys_mixed, 2)
    self.assertLen(ys_mixed[0], length)  # Should be unstacked
    np.testing.assert_array_equal(
        ys_mixed[1], jnp.array(expected_ys1)
    )  # Should be stacked

    for i in range(length):
      np.testing.assert_array_equal(ys_mixed[0][i], expected_ys0[i])

  def test_scan_with_unstacked_jitted(self):
    def body_fn(carry, x):
      return carry + x, (carry * 2, x * 3)

    init_carry = 0
    xs = jnp.arange(5)
    length = 5

    jitted_scan = jax.jit(
        instrument_jax_loop.scan_with_unstacked,
        static_argnames=('f', 'length', 'unstack'),
    )
    carry, _ = jitted_scan(
        f=body_fn, init=init_carry, xs=xs, length=length, unstack=True
    )
    self.assertEqual(carry, sum(xs))

if __name__ == '__main__':
  absltest.main()
