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

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from mishax import safe_greenlet
from mishax.examples import gemma
import numpy as np

gemma.install_clean()

from gemma import modules  # pylint: disable=g-import-not-at-top,g-bad-import-order
from gemma import transformer  # pylint: disable=g-bad-import-order


CONFIG = transformer.TransformerConfig(
    num_layers=2,
    num_embed=4,  # unused
    embed_dim=2,
    hidden_dim=12,  # unused
    num_heads=3,
    head_dim=4,
    num_kv_heads=3,
    max_cache_length=6,
    final_logit_softcap=None,
    attention_types=[modules.AttentionType.GLOBAL] * 2,
    use_post_attn_norm=False,
    use_post_ffw_norm=False,
)


class CallJitTransformerThrice(nn.Module):
  """Module to test avoiding JIT cache hits when running as generator.

  This is desirable because cache hits mean we never call safe_greenlet.yield_,
  so the tagged values in those calls are never exposed via the generator.

  The __call__ method calls the underlying jitted transformer 3 times; this is
  needed to get cache hits.
  """
  submodel: nn.jit(transformer.Transformer)

  @property
  def config(self):
    return self.submodel.config

  def setup(self):
    nn.share_scope(self, self.submodel)

  @nn.compact
  def __call__(self, *args, **kwargs):
    return [self.submodel(*args, **kwargs) for _ in range(3)]


MODEL = transformer.Transformer(CONFIG)
JIT_MODEL = nn.jit(transformer.Transformer)(CONFIG)
CALL_JIT_TRANSFORMER_THRICE = CallJitTransformerThrice(JIT_MODEL)
UNINSTRUMENTED_MODEL = gemma.TRANSFORMER_PATCHER.original_members[
    transformer.Transformer.__name__
](CONFIG)

BATCH_SIZE = 1
SEQ_SIZE = 4
TOKEN_INPUT = np.ones((BATCH_SIZE, SEQ_SIZE), dtype=np.int32)
ATTENTION_MASK = np.ones(
    (BATCH_SIZE, SEQ_SIZE, CONFIG.max_cache_length), dtype=np.bool
)


class GemmaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.empty_cache = CONFIG.init_cache(BATCH_SIZE, dtype=jnp.float32)
    self.positions = transformer.build_positions_from_mask(TOKEN_INPUT != 0)
    self.params = MODEL.init(
        jax.random.key(0),
        TOKEN_INPUT,
        self.positions,
        self.empty_cache,
        ATTENTION_MASK,
    )
    self.site_visits_if_once = []
    for site in gemma.Site:
      for layer in range(CONFIG.num_layers) if site.is_layer_site() else [None]:
        self.site_visits_if_once.append((layer, site))

  @parameterized.product(
      [
          dict(
              model=MODEL,
              visits_per_site=1,
              can_fetch_generated_values=True,
              mutable=True,
          ),
          dict(
              model=JIT_MODEL,
              visits_per_site=1,
              can_fetch_generated_values=False,
              mutable=True,
          ),
          dict(
              model=CALL_JIT_TRANSFORMER_THRICE,
              visits_per_site=3,
              can_fetch_generated_values=False,
              mutable=True,
          ),
          # Test with mutable=False to encourage jit cache hits.
          dict(
              model=CALL_JIT_TRANSFORMER_THRICE,
              visits_per_site=3,
              can_fetch_generated_values=False,
              mutable=False,
          ),
      ],
      [
          dict(use_cache=True, attn_mask=ATTENTION_MASK),
          dict(use_cache=False, attn_mask=ATTENTION_MASK[..., :SEQ_SIZE]),
      ],
  )
  @jax.numpy_rank_promotion('raise')
  @jax.checking_leaks()
  def test_iterate_through_activations_and_intervene(
      self,
      model,
      visits_per_site: int,
      can_fetch_generated_values: bool,
      use_cache: bool,
      mutable: bool,
      attn_mask: np.ndarray,
  ):
    with safe_greenlet.easy_greenlet(
        lambda: model.apply(
            self.params
            | gemma.vars_from_callback(gemma.GreenletYield(), model),
            TOKEN_INPUT,
            self.positions,
            self.empty_cache if use_cache else None,
            attn_mask,
            mutable=mutable,
        ),
    ) as run:

      keys_0 = None
      found_activations = []
      for layer, site, value in run:
        # We make 2 interventions: replacing layer 1 keys with layer 0 keys,
        # and replacing layer 1 attention outputs with zeros.
        match layer, site:
          case 0, (gemma.Site.KEYS):
            keys_0 = value
          case 1, (gemma.Site.KEYS):
            run.sendval = keys_0
          case 1, (gemma.Site.ATTN_OUTPUT_PRE_LINEAR):
            run.sendval = jnp.zeros_like(value)
        found_activations.append((layer, site))
    self.assertCountEqual(
        found_activations, self.site_visits_if_once * visits_per_site
    )

    if can_fetch_generated_values:
      keys_0 = jax.device_get(keys_0)
    else:
      with self.assertRaises(ReferenceError):
        keys_0 = jax.device_get(keys_0)

    if not mutable:
      self.assertNotIn(gemma.ACTIVATIONS, run.retval[1])
      return  # The rest of these checks rely on getting activations.

    acts = run.retval[1][gemma.ACTIVATIONS]
    for path, acts_for_site_and_layer in jax.tree.leaves_with_path(
        acts, is_leaf=lambda x: isinstance(x, tuple)
    ):
      self.assertLen(acts_for_site_and_layer, visits_per_site, path)

    if can_fetch_generated_values:
      np.testing.assert_array_equal(
          keys_0, acts['layer_0']['attn'][gemma.Site.KEYS][0]
      )

    acts_keys_0 = acts['layer_0']['attn'][gemma.Site.KEYS][0]
    acts_keys_1 = acts['layer_1']['attn'][gemma.Site.KEYS][0]
    np.testing.assert_array_equal(
        acts_keys_0,
        acts_keys_1,
        'Expected layer 1 keys to be replaced with layer 0 keys.',
    )

    acts_attn_output_1 = acts['layer_1'][gemma.Site.ATTN_OUTPUT][0]
    self.assertEqual(
        jnp.abs(acts_attn_output_1).max(),
        0,
        'Expected zero attention outputs in layer 1 after injecting'
        ' zeros before the linear attention output.',
    )

  def test_validate_non_pytree_callback(self):
    with self.assertRaises(ValueError):
      gemma.vars_from_callback(callback=lambda *_: None, model=MODEL)

  def test_validate_uninstrumented_model(self):
    with self.assertRaises(RuntimeError):
      gemma.vars_from_callback(
          callback=gemma.GreenletYield(), model=UNINSTRUMENTED_MODEL
      )


if __name__ == '__main__':
  absltest.main()
