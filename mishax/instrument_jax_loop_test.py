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
from absl.testing import parameterized
import jax
from mishax import instrument_jax_loop
import numpy as np


def body_fn(carry, x):
  return carry + x, (carry * 2, x * 2)


INIT = 3
XS = np.array([0, 1, 2])
LENGTH = 3
EXPECTED_CARRY = 3 + 0 + 1 + 2

TEST_CASES = (
    ('unstack_all', True, ((6, 0), (6, 2), (8, 4))),
    ('unstack_none', False, (np.array([6, 6, 8]), np.array([0, 2, 4]))),
    ('unstack_mixed', (True, False), ((6, 6, 8), np.array([0, 2, 4]))),
)


class InstrumentJaxLoopTest(parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  def test_scan_with_unstacked(self, unstack, expected_ys):
    carry, ys = (
        instrument_jax_loop.scan_with_unstacked(
            body_fn, INIT, XS, LENGTH, unstack=unstack
        )
    )
    self.assertEqual(carry, EXPECTED_CARRY)
    jax.tree.map(np.testing.assert_array_equal, ys, expected_ys)

  @parameterized.named_parameters(*TEST_CASES)
  def test_scan_with_unstacked_no_length(self, unstack, expected_ys):
    carry, ys = (
        instrument_jax_loop.scan_with_unstacked(
            body_fn, INIT, XS, unstack=unstack
        )
    )
    self.assertEqual(carry, EXPECTED_CARRY)
    jax.tree.map(np.testing.assert_array_equal, ys, expected_ys)

  @parameterized.named_parameters(*TEST_CASES)
  def test_scan_with_unstacked_jitted(self, unstack, expected_ys):
    jitted_scan = jax.jit(
        instrument_jax_loop.scan_with_unstacked,
        static_argnames=('f', 'length', 'unstack'),
    )
    carry, ys = jitted_scan(
        f=body_fn, init=INIT, xs=XS, length=LENGTH, unstack=unstack
    )
    self.assertEqual(carry, EXPECTED_CARRY)
    jax.tree.map(np.testing.assert_array_equal, ys, expected_ys)

if __name__ == '__main__':
  absltest.main()
