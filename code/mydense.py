# Dense layer adapted from the neural tangents library 
# we modified the init functions to return the prior density of the parameters when rng is None

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neural_tangents import stax
from typing import Optional

import enum
import functools
import operator as op
import string
from typing import Callable, Iterable, Optional, Sequence, Union
import warnings

import jax
from jax import eval_shape
from jax import lax
from jax import numpy as jnp
from jax import ops
from jax import random
from jax import ShapeDtypeStruct
from jax import vmap
from jax.core import ShapedArray
import jax.example_libraries.stax as ostax
import numpy as np

from neural_tangents._src.stax.requirements import Diagonal
from neural_tangents._src.stax.requirements import layer
from neural_tangents._src.stax.requirements import requires
from neural_tangents._src.stax.requirements import supports_masking
from neural_tangents._src.utils.kernel import Kernel
from neural_tangents._src.utils.typing import InternalLayerMasked

from neural_tangents._src.stax.linear import _affine


@layer
@supports_masking(remask_kernel=True)
def MyDense(
    out_dim: int,
    W_std: float = 1.,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  r"""Dense (fully-connected, matrix product).

  Based on :obj:`jax.example_libraries.stax.Dense`.

  Args:
    out_dim:
      The output feature / channel dimension. This is ignored in by the
      `kernel_fn` in `"ntk"` parameterization.

    W_std:
      Specifies the standard deviation of the weights.

    b_std:
      Specifies the standard deviation of the biases. `None` means no bias.

    batch_axis:
      Specifies which axis is contains different elements of the batch.
      Defaults to `0`, the leading axis.

    channel_axis: Specifies which axis contains the features / channels.
      Defaults to `-1`, the trailing axis. For `kernel_fn`, channel size is
      considered to be infinite.

    parameterization:
      Either `"ntk"` or `"standard"`.

      Under `"ntk"` parameterization (page 3 in "`Neural Tangent Kernel:
      Convergence and Generalization in Neural Networks
      <https://arxiv.org/abs/1806.07572>`_"),
      weights and biases are initialized as
      :math:`W_{ij} \sim \mathcal{N}(0,1)`, :math:`b_i \sim \mathcal{N}(0,1)`,
      and the finite width layer equation is
      :math:`z_i = \sigma_W / \sqrt{N} \sum_j W_{ij} x_j + \sigma_b b_i`, where
      `N` is `out_dim`.

      Under `"standard"` parameterization ("`On the infinite width limit of
      neural networks with a standard parameterization
      <https://arxiv.org/abs/2001.07301>`_".),
      weights and biases are initialized as :math:`W_{ij} \sim \mathcal{N}(0,
      W_{std}^2/N)`,
      :math:`b_i \sim \mathcal{N}(0,\sigma_b^2)`, and the finite width layer
      equation is
      :math:`z_i = \frac{1}{s} \sum_j W_{ij} x_j + b_i`, where `N` is `out_dim`.

      `N` corresponds to the respective variable in
      "`On the infinite width limit of neural networks with a standard
      parameterization <https://arxiv.org/abs/2001.07301>`_".

    s:
      only applicable when `parameterization="standard"`. A tuple of integers
      specifying the width scalings of the input and the output of the layer,
      i.e. the weight matrix `W` of the layer has shape
      `(s[0] * in_dim, s[1] * out_dim)`, and the bias has size `s[1] * out_dim`.

      .. note::
        We need `s[0]` (scaling of the previous layer) to infer `in_dim` from
        `input_shape`. Further, for the bottom layer, `s[0]` must be `1`, and
        for all other layers `s[0]` must be equal to `s[1]` of the previous
        layer. For the top layer, `s[1]` is expected to be `1` (recall that the
        output size is `s[1] * out_dim`, and in common infinite network
        research input and output sizes are considered fixed).

      `s` corresponds to the respective variable in
      "`On the infinite width limit of neural networks with a standard
      parameterization <https://arxiv.org/abs/2001.07301>`_".

      For `parameterization="ntk"`, or for standard, finite-width networks
      corresponding to He initialization, `s=(1, 1)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  # TODO(jaschasd): after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def _init_fn(rng, input_shape, out_dim):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    if rng is not None:
      rng1, rng2 = random.split(rng)
      W = random.normal(rng1, (input_shape[_channel_axis], out_dim))

      if b_std is None:
        b = None
      else:
        b_shape = [1] * len(input_shape)
        b_shape[channel_axis] = out_dim
        b = random.normal(rng2, b_shape)
      return output_shape, (W, b)
    else:
      return output_shape

  def ntk_init_fn(rng, input_shape, params=None):
    if rng is None: # eval prior density of params
      W, b = params
      W_mu = 0.
      W_sigma = W_std
      b_mu = 0.
      b_sigma = b_std

      output_shape = _init_fn(rng, input_shape, out_dim)

      logp = jnp.sum(jax.scipy.stats.norm.logpdf(W, loc=W_mu ,scale=W_sigma)) + jnp.sum(jax.scipy.stats.norm.logpdf(b,loc=b_mu ,scale=b_sigma)) 

      return output_shape, logp

    else:
      return _init_fn(rng, input_shape, out_dim)

  def standard_init_fn(rng, input_shape, params=None):

    W_sigma = W_std / (input_shape[channel_axis] / s[0])**0.5 
    W_mu = 0. 
    b_sigma = b_std
    b_mu = 0.
    
    if rng is None: # eval prior density of params
      W, b = params
      logp = jnp.sum(jax.scipy.stats.norm.logpdf(W, loc=W_mu ,scale=W_sigma)) + jnp.sum(jax.scipy.stats.norm.logpdf(b,loc=b_mu ,scale=b_sigma)) 
      output_shape = _init_fn(None, input_shape, out_dim * s[1])

      return output_shape, logp

    else:
      output_shape, (W, b) = _init_fn(rng, input_shape, out_dim * s[1])
      W *= W_sigma + W_mu
      b = None if b is None else b * b_sigma + b_mu
      return output_shape, (W, b)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError(f'Parameterization not supported: {parameterization}')

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    prod = jnp.moveaxis(jnp.tensordot(W, inputs, (0, channel_axis)),
                        0, channel_axis)

    if parameterization == 'ntk':
      norm = W_std / inputs.shape[channel_axis]**0.5
      outputs = norm * prod
      if b is not None:
        outputs += b_std * b
    elif parameterization == 'standard':
      outputs = prod / s[0]**0.5
      if b is not None:
        outputs += b
    else:
      raise ValueError(f'Parameterization not supported: {parameterization}')

    return outputs

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal())
  def kernel_fn(k: Kernel, **kwargs):
    """Compute the transformed kernels after a `Dense` layer."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
      if ntk is not None:
        ntk = nngp + W_std**2 * ntk
    elif parameterization == 'standard':
      input_width = k.shape1[channel_axis] / s[0]
      if ntk is not None:
        ntk = input_width * nngp + W_std**2 * ntk
        if b_std is not None:
          ntk += 1.
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=True,
                     is_input=False)

  def mask_fn(mask, input_shape):
    return jnp.all(mask, axis=channel_axis, keepdims=True)

  return init_fn, apply_fn, kernel_fn, mask_fn