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

"""Instrumented Gemma.

This demonstrates using Mishax to instrument an LLM codebase; it's pointed at
the https://github.com/google-deepmind/gemma reference implementation of Gemma.
"""
from collections.abc import Callable
import enum
import itertools
import typing
from typing import Any, Self, Iterable, TypeAlias
from flax import linen as nn
from flax import struct
from flax import traverse_util
import jax
from mishax import ast_patcher
from mishax import safe_greenlet

if typing.TYPE_CHECKING:
  from gemma import transformer  # pylint: disable=g-bad-import-order


# Flax collection names.
ACTIVATIONS = 'activations'
CALLBACK = 'callback'


@jax.tree_util.register_static
class Site(enum.StrEnum):
  """Instrumentation site within a Gemma forward pass.

  Each specifies in site.path_from_block what its module path is from the
  surrounding Block for the layer, or None if it isn't a layer site.
  """
  ATTN_INPUT_PRE_LAYERNORM = enum.auto()
  PRE_ATTN_LAYERNORM = enum.auto()
  KEYS = enum.auto(), 'attn'
  VALUES = enum.auto(), 'attn'
  QUERIES = enum.auto(), 'attn'
  ATTN_LOGITS = enum.auto(), 'attn'
  ATTNS = enum.auto(), 'attn'
  ATTN_OUTPUT_PRE_LINEAR = enum.auto(), 'attn'
  ATTN_OUTPUT = enum.auto()
  POST_ATTN_RESIDUAL = enum.auto()
  PRE_MLP_LAYERNORM = enum.auto()
  MLP_HIDDEN = enum.auto(), 'mlp'
  MLP_HIDDEN_PRE_GATE = enum.auto(), 'mlp'
  MLP_POST_ACTIVATION = enum.auto(), 'mlp'
  MLP_OUTPUT = enum.auto()
  FINAL_RESIDUAL = enum.auto()
  INPUTS = enum.auto(), None
  FINAL_RESIDUAL_POST_LAYERNORM = enum.auto(), None

  def __new__(cls, value, *path_from_block):
    retval = str.__new__(cls, value)
    retval._value_ = value
    if path_from_block == (None,):
      path_from_block = None
    retval.path_from_block = path_from_block
    return retval

  def is_layer_site(self) -> bool:
    return self.path_from_block is not None


def _tag(
    module: nn.Module, site: Site, value: jax.Array, /
) -> jax.Array | None:
  if callback := module.get_variable(CALLBACK, site, None):
    if (result := callback(value)) is not None:
      value = result
  if not module.is_initializing():
    module.sow(ACTIVATIONS, site, value)
  return value


PREFIX = f"""
from {__name__} import Site
from {__name__} import _tag as tag
"""

MODULES_PATCHER = ast_patcher.ModuleASTPatcher(
    'gemma.modules',
    ast_patcher.PatchSettings(prefix=PREFIX),
    Attention=[
        # attn_output_pre_linear
        'attn_output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)',
        """
        encoded = tag(self, Site.ATTN_OUTPUT_PRE_LINEAR, encoded)
        attn_output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)
        """,
        # keys, values
        'key_proj = positional_embeddings.apply_rope(key_proj, segment_pos)',
        """
        key_proj = positional_embeddings.apply_rope(key_proj, segment_pos)
        key_proj = tag(self, Site.KEYS, key_proj)
        value_proj = tag(self, Site.VALUES, value_proj)
        """,
        # queries and attn_logits when use_gqa
        'logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_scaled, key_proj)',
        """
        query_scaled = tag(self, Site.QUERIES, query_scaled)
        logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_scaled, key_proj)
        logits = tag(self, Site.ATTN_LOGITS, logits)
        """,
        # queries and attn_logits when not use_gqa
        'logits = jnp.einsum("BTNH,BSNH->BTNS", query_scaled, key_proj)',
        """
        query_scaled = tag(self, Site.QUERIES, query_scaled)
        logits = jnp.einsum("BTNH,BSNH->BTNS", query_scaled, key_proj)
        logits = tag(self, Site.ATTN_LOGITS, logits)
        """,
        # attns,
        'probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)',
        """
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
        probs = tag(self, Site.ATTNS, probs)
        """,
    ],
    FeedForward=[
        # mlp_hidden
        'nn.gelu(ff_gate)',
        'nn.gelu(tag(self, Site.MLP_HIDDEN, ff_gate))',
        # mlp_hidden_pre_gate, mlp_post_activation
        'activations = gate_value * ff1',
        """
        ff1 = tag(self, Site.MLP_HIDDEN_PRE_GATE, ff1)
        activations = tag(self, Site.MLP_POST_ACTIVATION, gate_value * ff1)
        """,
    ],
    Block=[
        # attn_input, pre_attn_layernorm
        'inputs_normalized = self.pre_attention_norm(x)',
        """
        x = tag(self, Site.ATTN_INPUT_PRE_LAYERNORM, x)
        inputs_normalized = tag(
            self, Site.PRE_ATTN_LAYERNORM, self.pre_attention_norm(x)
        )
        """,
        # attn_output, post_attn_residual
        'attn_output += x',
        """
        attn_output = tag(self, Site.ATTN_OUTPUT, attn_output)
        attn_output += x
        attn_output = tag(self, Site.POST_ATTN_RESIDUAL, attn_output)
        """,
        # pre_mlp_layernorm
        'self.pre_ffw_norm(attn_output)',
        'tag(self, Site.PRE_MLP_LAYERNORM, self.pre_ffw_norm(attn_output))',
        # mlp_output
        'self.mlp(outputs)',
        'tag(self, Site.MLP_OUTPUT, self.mlp(outputs))',
        # final_residual
        'outputs += attn_output',
        'outputs = tag(self, Site.FINAL_RESIDUAL, outputs + attn_output)',
    ],
)
TRANSFORMER_PATCHER = ast_patcher.ModuleASTPatcher(
    'gemma.transformer',
    ast_patcher.PatchSettings(prefix=PREFIX),
    Transformer=[
        # inputs
        'self.embedder.encode(last_tokens)',
        'tag(self, Site.INPUTS, self.embedder.encode(last_tokens))',
        # final_residual_post_layernorm
        'self.final_norm(x)',
        'tag(self, Site.FINAL_RESIDUAL_POST_LAYERNORM, self.final_norm(x))',
    ],
)


def install():
  """Installs the patchers, so that all new modules will be instrumented."""
  MODULES_PATCHER.install()
  TRANSFORMER_PATCHER.install()


def install_clean():
  """Same as install, but ensuring fresh import of the patched modules."""
  MODULES_PATCHER.install_clean()
  TRANSFORMER_PATCHER.install_clean()


# Callback to be called at each layer and site in the Transformer.
# The return value will be substituted for the original value at the site,
# unless it's None.
Callback: TypeAlias = Callable[[int | None, Site, jax.Array], jax.Array | None]


class _SelectedCallback(struct.PyTreeNode):
  """Callback specialized to one layer and site."""
  callback: Callback = struct.field()
  layer: int | None = struct.field(pytree_node=False)
  site: Site = struct.field()

  def __call__(self, value: jax.Array):
    return self.callback(self.layer, self.site, value)


def _validate_model(model: 'transformer.Transformer'):
  if not (
      TRANSFORMER_PATCHER.is_installed() and MODULES_PATCHER.is_installed()
  ) or isinstance(model, TRANSFORMER_PATCHER.original_members['Transformer']):
    raise RuntimeError(
        'Please install the patchers before constructing the transformer.'
    )


def _validate_callback(callback: Callback):
  if not isinstance(callback, Callable):
    raise ValueError('Callback must be a callable.')
  if not all(
      isinstance(leaf, jax.typing.ArrayLike)
      for leaf in jax.tree.leaves(callback)
  ):
    raise ValueError(
        'Callback must be a pytree with ArrayLike leaves. Consider using'
        ' `jax.tree_util.Partial` or `flax.struct`.'
    )


def vars_from_callback(callback: Callback, model: 'transformer.Transformer'):
  """Returns variables to be passed to the model, to invoke the callback."""
  _validate_model(model)
  _validate_callback(callback)
  flat_retval = {}
  for site in Site:
    if site.is_layer_site():
      for layer in range(model.config.num_layers):
        flat_retval[CALLBACK, f'layer_{layer}', *site.path_from_block, site] = (
            _SelectedCallback(callback, layer, site)
        )
    else:
      flat_retval[CALLBACK, site] = _SelectedCallback(callback, None, site)
  return traverse_util.unflatten_dict(flat_retval)


@jax.tree_util.register_pytree_with_keys_class
class GreenletYield(Callback):
  """Callback that yields values to parent greenlet, with layer and site info.

  This class is registered as a pytree in such a way that the treedef is
  different every time `flatten` is called, to avoid caching.

  NOTE: If the callback is called with a jax.core.Tracer then this can produce
  tracer leaks; these can be debugged using `jax.checking_leaks()`, but that may
  find false positives if using easy_greenlet, via the loop variable. To fix
  these, `del` the activation in the loop variable after using it in the loop.
  """

  def __call__(self, layer: int | None, site: Site, value: jax.Array):
    return safe_greenlet.yield_(layer, site, value)

  def __init__(self):
    self._counter = itertools.count()

  def tree_flatten_with_keys(
      self,
  ) -> tuple[Iterable[tuple[jax.tree_util.KeyEntry, Any]], tuple[Self, int]]:
    # Ensure the treedef is different every time flatten is called.
    return ((), (self, next(self._counter)))

  @classmethod
  def tree_unflatten(
      cls, aux_data: tuple[Self, int], children: Iterable[Any]
  ) -> Self:
    del children
    return aux_data[0]
