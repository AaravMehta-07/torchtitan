# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config helpers shared by the RL recipes."""

from collections.abc import Callable
from typing import Any

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig


def build_rl_model_configs(
    build_model_config: Callable[..., Decoder.Config],
    flavor: str,
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    **kwargs: Any,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Build the ``(trainer, generator)`` model configs of an RL recipe.

    Both come from the same ``build_model_config(flavor, **kwargs)`` call, so
    they share the architecture; each is sharded for its own role's
    parallelism. Apply each role's complete transform list in the recipe after
    setting its options. Changes made to one model must be made to the other.

    RL logprob / KL math needs fp32 lm_head logits. Recipes apply
    ``LMHeadFP32OutputTransform`` to both roles to preserve that precision in
    the vocabulary projection; this helper only builds the model configs.
    """
    return (
        build_model_config(flavor, parallelism=trainer_parallelism, **kwargs),
        build_model_config(
            flavor, parallelism=generator_parallelism.to_training(), **kwargs
        ),
    )
