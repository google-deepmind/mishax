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
from flax import linen as nn
import jax
from mishax import instrument_flax_loop
import numpy as np


NUM_LAYERS = 6


class Foo(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(4)(x)
    if (loop_index := self.get_variable('loop', 'index')) is not None:
      for i in range(NUM_LAYERS):
        instrument_flax_loop.sow(
            self, 'activations', f'layer_{i}', x, i == loop_index
        )
    return x, None


class InstrumentFlaxScanTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.scanned = instrument_flax_loop.instrumented_scan(
        Foo,
        variable_zero_init_carry='activations',
        length=6,
        split_rngs=dict(params=True),
        variable_axes=dict(params=0),
        unroll=2,
    )()
    self.input = np.ones(4)
    self.variables = self.scanned.init(
        jax.random.key(0), self.input, mutable='params'
    )
    (self.scan_out, _), variables_out = self.scanned.apply(
        self.variables, self.input, mutable='activations'
    )
    self.activations = variables_out['activations']

  def test_can_run_without_acts(self):
    (scan_out_no_acts, _) = self.scanned.apply(self.variables, self.input)
    np.testing.assert_allclose(scan_out_no_acts, self.scan_out, atol=1e-7)

  def test_acts_are_correct(self):
    np.testing.assert_allclose(
        self.scan_out,
        self.activations[f'layer_{NUM_LAYERS - 1}'],
        atol=1e-7,
    )
    x = self.input
    for i in range(NUM_LAYERS):
      vars_i = jax.tree.map(lambda x, i=i: x[i], self.variables)
      x, _ = Foo().apply(vars_i, x)
      np.testing.assert_allclose(self.activations[f'layer_{i}'], x, atol=1e-7)


if __name__ == "__main__":
  absltest.main()
