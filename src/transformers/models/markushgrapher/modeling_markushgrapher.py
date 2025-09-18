# coding=utf-8
# Copyright 2023 Microsoft Research and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch UDOP model."""

import collections
import logging
import math
from types import SimpleNamespace
import os
import matplotlib.pyplot as plt
from pprint import pprint
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

from transformers import MarkushgrapherConfig
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from molscribe.model import Encoder, Decoder
from molscribe.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

MARKUSHGRAPHER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # TODO update organization
    "nielsr/udop-large",
    # See all UDOP models at https://huggingface.co/models?filter=udop
]


_CONFIG_FOR_DOC = "MarkushgrapherConfig"


MARKUSHGRAPHER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`UdopConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MARKUSHGRAPHER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. UDOP is a model with relative position embeddings so
            you should be able to pad the inputs on both the right and the left. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for detail.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) T5 uses the `pad_token_id` as the starting
            token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
            `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
            `decoder_input_ids` for pretraining take a look at [T5 Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix. If
            `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of
            `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


MARKUSHGRAPHER_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BaseModelOutputWithAttentionMask(ModelOutput):
    """
    Class for the model's outputs that may also contain a past key/values (to speed up sequential decoding). Includes
    an additional attention mask.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model. If `past_key_values` is used only
            the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
        when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`. Contains pre-computed hidden-states (key and values in the
            self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and
        `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_visual_bbox(image_size=224, patch_size=16):
    image_feature_pool_shape = [image_size // patch_size, image_size // patch_size]
    visual_bbox_x = (
        torch.arange(
            0,
            1.0 * (image_feature_pool_shape[1] + 1),
            1.0,
        )
        / image_feature_pool_shape[1]
    )
    visual_bbox_y = (
        torch.arange(
            0,
            1.0 * (image_feature_pool_shape[0] + 1),
            1.0,
        )
        / image_feature_pool_shape[0]
    )
    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
        ],
        dim=-1,
    ).view(-1, 4)
    return visual_bbox_input


def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


# Markushgrapher
def combine_image_text_embeddings(
    image_embeddings,
    inputs_embeds,
    bbox,
    visual_bbox,
    attention_mask=None,
    num_patches=14,
    max_len=0,
    image_size=224,
    patch_size=16,
    verbose=False,
):
    """
    Combine the image and text embeddings for the input to the encoder/decoder of Markushgrapher.

    Conclusion:
    The output sequence is following this schema:
        - Text:
             - Prompt text tokens
             - OCR text tokens summed with the visual patches representing them
             - Padding (up to 512)
         - Image:
             - Remaining visual patches (that do not represent any OCR text token)
             - Padding (up to 1024)
    The output sequence length is 1536.
    
    Warning: Bbox needs to be between 0 and 1.
    """
    if verbose:
        print("image_size", image_size)  # 512
        print("patch_size", patch_size)  # 16
        print("num_patches", num_patches)  # 32
        print("bbox size()", bbox.size())
        print("bbox", bbox)

    sequence_length = num_patches

    # Determine the central point of each text token's bounding box and map it to the corresponding image patch index.
    ocr_points_x = torch.clip(
        torch.floor((bbox[:, :, 0] + bbox[:, :, 2]) / 2.0 * sequence_length).long(), 0, sequence_length - 1
    )
    ocr_points_y = (
        torch.clip(torch.floor((bbox[:, :, 1] + bbox[:, :, 3]) / 2.0 * sequence_length).long(), 0, sequence_length - 1)
        * sequence_length
    )
    ocr_points = ocr_points_x + ocr_points_y
    if verbose:
        print("ocr_points size", ocr_points.size())
        print("ocr_points", ocr_points)
    # Identify bounding boxes that are either all zeros or all ones, possibly representing special cases or invalid regions.
    bbox = bbox.float()
    target_seg = (bbox.mean(-1) == 0.0) | (bbox.mean(-1) == 1.0)

    # Extract image embeddings corresponding to text tokens and integrate them into the text embeddings.
    repeated_vision_embeds = torch.gather(
        image_embeddings, 1, ocr_points.unsqueeze(-1).repeat(1, 1, image_embeddings.size(-1))
    )
    repeated_vision_embeds[target_seg] = 0.0
    if verbose:
        print("repeated_vision_embeds", repeated_vision_embeds.size())
    inputs_embeds += repeated_vision_embeds  # lum: Add visual tokens to text tokens

    # Identify which image patches are not associated with any text tokens to be processed separately.
    patch_inds = torch.full_like(image_embeddings[:, :, 0], True).bool()
    ind = torch.cat(
        [
            torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
            ocr_points[:, :, None],
        ],
        -1,
    ).flatten(0, 1)
    rows, cols = zip(*ind)
    if verbose:
        print("rows", len(rows))  # 0 values
        print("cols", len(cols))  # 0...0 1023...1023 values
    patch_inds[rows, cols] = False

    # Collect image embeddings that are not tied to any text tokens for separate processing.
    if verbose:
        print("image_embeddings", image_embeddings.size())
        print("patch_inds size()", patch_inds.size())
        print("patch_inds len()", len(patch_inds))  # 1
        print("patch_inds", patch_inds)
    input_vision_patches = [
        image_embeddings[i][patch_inds[i]] for i in range(len(patch_inds))
    ]  # lum: Get a list (batch size) with a torch having the unused image patches only

    # Ensure that visual_bbox corresponds only to the selected image patches.
    if visual_bbox is None:
        visual_bbox = get_visual_bbox(image_size=image_size, patch_size=patch_size)
        visual_bbox = visual_bbox.unsqueeze(0).repeat(image_embeddings.size(0), 1, 1)
        visual_bbox = visual_bbox.to(image_embeddings.device)
    if verbose:
        print("visual_bbox size A", visual_bbox.size())
    visual_bbox = [visual_bbox[i][patch_inds[i]] for i in range(len(patch_inds))]
    if verbose:
        print("visual_bbox size B", visual_bbox[0].size())
    # Generate an attention mask for the visual patches, indicating which patches should be attended to.
    # Proper handling of attention masks ensures that padded tokens or irrelevant patches do not influence the model's attention mechanisms.
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_bbox]

    # Set the maximum sequence length for padding purposes.
    if max_len == 0:
        max_len = image_embeddings.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)

    # Ensure that all vision-related tensors have consistent dimensions by padding them to max_len.
    if verbose:
        print("input_vision_patches", [item.size() for item in input_vision_patches])  # [955, 4]
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(image_embeddings[0, 0])) for item in input_vision_patches]
    )
    if verbose:
        print("inputs_vision_patches", inputs_vision_patches.size())  # [1, 1024, 4]
        print("visual_bbox size 1", visual_bbox[0].size())
    visual_bbox = torch.stack([pad_sequence(item, max_len, torch.zeros_like(bbox[0, 0])) for item in visual_bbox])
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask]
        )

    # if vis_special_token is not None:
    #     inputs_vision_patches += vis_special_token

    # Combine the enriched text embeddings with the additional visual patches, updating bounding boxes and attention masks accordingly.
    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    bbox = torch.cat([bbox, visual_bbox], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)

    # print([i for i, v in enumerate(attention_mask[0].tolist()) if v == 0])
    # print(attention_mask.size())
    if verbose:
        print("visual_bbox size 2", visual_bbox.size())
    return inputs_embeds, bbox, attention_mask


class MarkushgrapherPatchEmbeddings(nn.Module):
    """2D Image to Patch Embeddings"""

    # Markushgrapher
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self._config = config
        # Notes:
        # in_channel, out_channel, kernel_size, stride
        # 3, hidden_size, kernel_size = patch_size, stride = patch_size
        # output size = (input size − kernel size + 2×padding)/stride + 1
        if config.architecture_variant != "vision-encoder":
            self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        if config.architecture_variant == "vision-encoder":
            self.encoder = models.resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.encoder_num_channels = 512
            self.encoder_projection = nn.Linear(self.encoder_num_channels, hidden_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f" ({self.image_size[0]}*{self.image_size[1]})."
            )
        if self._config.architecture_variant != "vision-encoder":
            embeddings = self.proj(pixel_values)
            embeddings = embeddings.flatten(2).transpose(1, 2)

        if self._config.architecture_variant == "vision-encoder":
            resnet_features = self.encoder(pixel_values)
            # ResNet downscales the spatial dimensions, so we need to upsample it
            target_height = self.image_size[0] // self.patch_size[0]
            target_width = self.image_size[1] // self.patch_size[1]
            resnet_features_upsampled = F.interpolate(
                resnet_features, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
            resnet_features_upsampled = resnet_features_upsampled.flatten(2).transpose(1, 2)
            embeddings = self.encoder_projection(resnet_features_upsampled)

        return embeddings


class MarkushgrapherPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. Based on `T5PreTrainedModel`.
    """

    config_class = MarkushgrapherConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MarkushgrapherBlock"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, MarkushgrapherLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=factor).to(
                module.weight.dtype
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, MarkushgrapherModel):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, MarkushgrapherForConditionalGeneration):
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, MarkushgrapherDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MarkushgrapherDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MarkushgrapherAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MarkushgrapherAttention, MarkushgrapherStack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In Markushgrapher it is usually set to the pad_token_id."
            " See Markushgrapher docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Udop
class MarkushgrapherLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the Udop style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Udop uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    MarkushgrapherLayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of MarkushgrapherLayerNorm")
except ImportError:
    # using the normal MarkushgrapherLayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to MarkushgrapherLayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(MarkushgrapherLayerNorm)


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->Udop
class MarkushgrapherDenseActDense(nn.Module):
    def __init__(self, config: MarkushgrapherConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Markushgrapher
class MarkushgrapherDenseGatedActDense(nn.Module):
    def __init__(self, config: MarkushgrapherConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->Udop
class MarkushgrapherLayerFF(nn.Module):
    def __init__(self, config: MarkushgrapherConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = MarkushgrapherDenseGatedActDense(config)
        else:
            self.DenseReluDense = MarkushgrapherDenseActDense(config)

        self.layer_norm = MarkushgrapherLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->Udop
class MarkushgrapherAttention(nn.Module):
    def __init__(self, config: MarkushgrapherConfig, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->Udop
class MarkushgrapherLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = MarkushgrapherAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = MarkushgrapherLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->Udop
class MarkushgrapherLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = MarkushgrapherAttention(config, has_relative_attention_bias=False)
        self.layer_norm = MarkushgrapherLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->Udop
class MarkushgrapherBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(MarkushgrapherLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MarkushgrapherLayerCrossAttention(config))

        self.layer.append(MarkushgrapherLayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class MarkushgrapherCellEmbeddings(nn.Module):
    def __init__(self, max_2d_position_embeddings=501, hidden_size=1024, ccat=False):
        super(MarkushgrapherCellEmbeddings, self).__init__()
        self.ccat = ccat
        self.max_2d_position_embeddings = max_2d_position_embeddings
        if ccat:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
        else:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)

    def forward(self, bbox):
        bbox = torch.clip(bbox, 0.0, 1.0)
        bbox = (bbox * (self.max_2d_position_embeddings - 1)).long()
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        if self.ccat:
            embeddings = torch.cat(
                [
                    left_position_embeddings,
                    upper_position_embeddings,
                    right_position_embeddings,
                    lower_position_embeddings,
                ],
                dim=-1,
            )
        else:
            embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
            )

        return embeddings


# get function for bucket computation
# protected member access seems to be lesser evil than copy paste whole function
get_relative_position_bucket = MarkushgrapherAttention._relative_position_bucket
AUGMENTATION_RANGE = (0.80, 1.25)


class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases :param num_heads: number of heads in lm model, it will create embeddings of size
    `num_heads`,
        which will be added to scores per each token pair
    :param relative_attention_num_buckets: pair token metric
        (distance in the sequence, distance in pixels etc.) will be bucketed, parameter is defining number of such
        buckets
    :param bidirectional: defining if for pair of tokens distance should be bidirecional,
        if bidirectional=False, then distance(tok1, tok2) == distance(tok2, tok1)
    :param scaling_factor: defining factor which will be used to scale relative distance :param max_distance: all
    distances above this value will end up in the one/same bucket :param augmentation: whether to multiple relative
    distances by random scalar :param expand: used for re-using pretrained model with subsequent addition of
    prefix_bucket
    """

    def __init__(
        self,
        num_heads=None,
        relative_attention_num_buckets=32,
        bidirectional=True,
        scaling_factor=1,
        max_distance=128,
        level="tokens",
        augmentation=False,
        prefix_bucket=False,
        expand=False,
    ):
        super(RelativePositionBiasBase, self).__init__()
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and not self.expand else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Optional[Tensor] = None,
        bbox: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        pass

    def get_bucket(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        relative_position = self.prepare_input(attention_mask, bbox)
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)

    def forward(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # re-using pretrained model with subsequent addition of prefix_bucket
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
            new_bias.weight.data[: self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets :] = 0.1
            self.relative_attention_bias = new_bias
            self.expand = False

        rp_bucket = self.get_bucket(attention_mask, bbox)

        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
                rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
            # based on assumption that prefix bboxes are negative
            is_prefix = bbox[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

        values: Tensor = self.relative_attention_bias(rp_bucket)
        if values.dim() != 4:
            raise ValueError("Wrong dimension of values tensor")
        values = values.permute([0, 3, 1, 2])

        return values


class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is their distance in the sequence.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if self.scaling_factor != 1:
            raise ValueError("No need to scale 1d features")
        relative_position = self.get_relative_position(
            torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)[None, :]
        )

        return relative_position


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as there are in small (0,1) range")
        if bbox is None:
            raise ValueError("Bbox is required for horizontal relative position bias")
        # get x positions of left point of bbox
        horizontal_position: Tensor = bbox[:, :, [0, 2]].mean(dim=-1)

        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as there are in small (0,1) range")
        if bbox is None:
            raise ValueError("Bbox is required for vertical relative position bias")
        # get y positions of middle of bbox
        vertical_position: Tensor = bbox[:, :, [1, 3]].mean(dim=-1)

        return self.get_relative_position(vertical_position)


class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class which sums up various computed biases.

        Args:
            modules (Sequence[RelativePositionBiasBase]):
                List of relative bias modules.
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(
        self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None
    ) -> Union[float, Tensor]:
        x = 0.0
        for bias in self.biases:  # type: ignore
            x = bias(attention_mask, bbox) + x

        return x


BIAS_CLASSES = {
    "1d": RelativePositionBias1D,
    "horizontal": RelativePositionBiasHorizontal,
    "vertical": RelativePositionBiasVertical,
}


def create_relative_bias(config: MarkushgrapherConfig) -> Sequence[RelativePositionBiasBase]:
    """
    Creates empty list or one/multiple relative biases.

    :param config: Model's configuration :return: Sequence with created bias modules.
    """
    bias_list = []
    if hasattr(config, "relative_bias_args"):
        for bias_kwargs_org in config.relative_bias_args:
            bias_kwargs = deepcopy(bias_kwargs_org)
            bias_type = bias_kwargs.pop("type")
            model_num_heads = config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
            if "num_heads" in bias_kwargs:
                if bias_kwargs["num_heads"] != model_num_heads:
                    raise ValueError("Number of heads must match num of heads in the model")
            else:
                bias_kwargs["num_heads"] = model_num_heads
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))  # type: ignore

    return bias_list


class MarkushgrapherStack(MarkushgrapherPreTrainedModel):
    """
    This class is based on `T5Stack`, but modified to take into account the image modality as well as 2D position
    embeddings.
    """

    def __init__(self, config, embed_tokens=None, embed_patches=None):
        super().__init__(config)

        print("MarkushgrapherStack self.config.architecture_variant:", self.config.architecture_variant)
        self.embed_tokens = embed_tokens
        self.embed_patches = embed_patches
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length
        self.num_layers = config.num_layers

        self.block = nn.ModuleList(
            [MarkushgrapherBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = MarkushgrapherLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = MarkushgrapherCellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        # Markushgrapher
        if ("molscribe-encoder-4" in self.config.architecture_variant) or (
            "molscribe-encoder-6" in self.config.architecture_variant
        ):
            molscribe_config = SimpleNamespace(
                **{
                    "encoder": "swin_base",
                    "use_checkpoint": True,
                    "formats": ["chartok_coords", "edges"],
                    "vocab_file": os.path.dirname(__file__)
                    + "/../../../../../MolScribe/molscribe/vocab/vocab_chars.json",
                    "coord_bins": 64,
                    "sep_xy": True,
                    "continuous_coords": False,
                    "embed_dim": 256,
                    "enc_pos_emb": False,
                    "decoder_dim": 512,
                    "decoder_layer": 1,
                    "attention_dim": 256,
                    "dec_num_layers": 6,
                    "dec_hidden_size": 256,
                    "dec_attn_heads": 8,
                    "dec_num_queries": 128,
                    "hidden_dropout": 0.1,
                    "attn_dropout": 0.1,
                    "max_relative_positions": 0,
                    "beam_size": 1,
                    "n_best": 1,
                    "predict_coords": False,
                    "save_attns": False,
                    "molblock": False,
                    "compute_confidence": False,
                    "keep_main_molecule": False,
                }
            )
            self.molscribe_encoder = Encoder(molscribe_config, pretrained=False)
            # molscribe_config.encoder_dim = self.molscribe_encoder.n_features
            # self.molscribe_decoder = Decoder(molscribe_config, get_tokenizer(molscribe_config))

        if "molscribe-encoder-6" in self.config.architecture_variant:
            self.molscribe_projector = nn.Sequential(
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 1024)
            )

    @staticmethod
    def _get_relative_bias(config: MarkushgrapherConfig) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # Markushgrapher
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        head_mask=None,
        past_key_values=None,
        ids_keep=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cross_attn_head_mask=None,
        position_bias=None,  # modified line,
        image_embeddings=None,  # modified line,
        bbox=None,  # modified line,
        visual_bbox=None,  # modified line,
        num_patches=None,  # modified line,
        special_vis_token=None,
        verbose=False,
    ):
        # Debugging
        # if (pixel_values is not None):
        #     save_image(pixel_values[0], f"{input_ids.tolist()[0][len(input_ids.tolist()[0])-5:]}.png")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input embeddings processing
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            bbox = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning("Empty batch")
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to intialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeddings = self.embed_patches(pixel_values)

        if image_embeddings is not None:
            if verbose:
                print("bbox size before combine_image_text_embeddings", bbox.size())
                print("inputs_embeds size before combine_image_text_embeddings", inputs_embeds.size())

            # Combine visual and OCR text embeddings
            if num_patches is None:
                num_patches = self.config.image_size // self.config.patch_size
            inputs_embeds, bbox, attention_mask = combine_image_text_embeddings(
                image_embeddings,
                inputs_embeds,
                bbox,
                visual_bbox,
                attention_mask,
                num_patches,
                0,
                self.config.image_size,
                self.config.patch_size,
            )

            input_shape = inputs_embeds.size()[:-1]
            if verbose:
                print("Image and text embeddings combination")
                print(
                    "inputs_embeds size:", inputs_embeds.size()
                )  # [1, 1063, 1024], [1, 1109, 1024] or  changes from one input to the other
                print("bbox:", bbox.tolist())
                print("bbox size:", bbox.size())
                print(
                    "bbox valid indices:",
                    [i for i, b in enumerate(bbox.tolist()[0]) if b == [1000.0, 1000.0, 1000.0, 1000.0]],
                )
                print(
                    "bbox filtered size:", len([b for b in bbox.tolist()[0] if b != [0.0, 0.0, 0.0, 0.0]])
                )  # 1094 for inputs_embeds.size() = [1, 1109, 1024]
                print("bbox size after combine_image_text_embeddings", bbox.size())
                print("inputs_embeds size after combine_image_text_embeddings", inputs_embeds.size())
            """
            *bbox*
            [0., 0., 0., 0.],
            ...                               (length of the question times) (13)
            [0., 0., 0., 0.],
            [1000.0, 1000.0, 1000.0, 1000.0], 
            [197.0, 279.0, 214.0, 294.0],
            ...                               (length of OCR boxes input)
            [292.0, 383.0, 297.0, 396.0], 
            [1000.0, 1000.0, 1000.0, 1000.0],
            [0.03125, 0.0, 0.0625, 0.03125],
            ...                               (length of the remaining visual embeddings)
            [0.9375, 0.9688, 0.9688, 1.0000]
            [0., 0., 0., 0.]                  (2 zero padding)
            [0., 0., 0., 0.]
            """
            # Note: conclusion: indices of group definitions are the same than in inputs_embeds.

        if not self.is_decoder and bbox is not None:
            inputs_embeds += self.cell2dembedding(bbox)

        # Add MolScribe encoding
        if (pixel_values is not None) and (self.config.architecture_variant == "molscribe-encoder-4"):
            if verbose:
                print("MolScribe encoder parameters")
                for i, (name, param) in enumerate(self.molscribe_encoder.named_parameters()):
                    if i != 0:
                        continue
                    print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")
                print("Molscribe decoder parameters")
                for i, (name, param) in enumerate(self.molscribe_decoder.named_parameters()):
                    if i != 0:
                        continue
                    print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")

            # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
            pixel_values_resized = torch.stack(
                [
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                    for img in pixel_values
                ]
            )
            if verbose:
                print(pixel_values_resized)
                image_np = pixel_values_resized[0].permute(1, 2, 0).cpu().numpy()
                plt.imshow(image_np)
                plt.savefig("test2.png")

            # Get MolScribe features
            molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)

            # Sum
            inputs_embeds[:, (inputs_embeds.size(1) - molscribe_features.size(1)) :, :] += molscribe_features
            if verbose:
                print(self.molscribe_decoder.decode(molscribe_features, hiddens, refs=None))

        if (pixel_values is not None) and (self.config.architecture_variant == "molscribe-encoder-6"):
            # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
            pixel_values_resized = torch.stack(
                [
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                    for img in pixel_values
                ]
            )

            # Get MolScribe features
            molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)
            molscribe_features_projected = self.molscribe_projector(molscribe_features)

            # Concatenate
            # inputs_embeds[:, (inputs_embeds.size(1) - molscribe_features.size(1)):, :] += molscribe_features
            inputs_embeds = torch.cat((molscribe_features_projected, inputs_embeds), dim=1)
            attention_mask = torch.cat(
                (
                    torch.ones(attention_mask.size(0), molscribe_features_projected.size(1)).to(self.device),
                    attention_mask,
                ),
                dim=1,
            )
            bbox = torch.cat(
                (
                    torch.zeros(bbox.size(0), molscribe_features_projected.size(1), bbox.size(2)).to(self.device),
                    bbox,
                ),
                dim=1,
            )
            input_shape = inputs_embeds.size()[:-1]
            if verbose:
                print(self.molscribe_decoder.decode(molscribe_features, hiddens, refs=None))

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(attention_mask=attention_mask, bbox=bbox)
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    attention_mask,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithAttentionMask(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class MarkushgrapherModel(MarkushgrapherPreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    """ """

    def __init__(self, config):
        super(MarkushgrapherModel, self).__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = MarkushgrapherPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MarkushgrapherStack(encoder_config, self.shared, self.patch_embed)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MarkushgrapherStack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(MARKUSHGRAPHER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        bbox: Dict[str, Any] = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, Markushgrapher
        >>> from huggingface_hub import hf_hub_download
        >>> from PIL import Image
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("nielsr/udop-large")
        >>> model = MarkushgrapherModel.from_pretrained("nielsr/udop-large")

        >>> # load document image
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> # prepare for the model
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

        >>> # forward pass
        >>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1, 1024]
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                bbox=bbox,
                visual_bbox=visual_bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            # we filter out the attention mask
            decoder_outputs = tuple(value for idx, value in enumerate(decoder_outputs) if idx != 1)
            encoder_outputs = tuple(value for idx, value in enumerate(encoder_outputs) if idx != 1)
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The UDOP encoder-decoder Transformer with a language modeling head on top, enabling to generate text given document
    images and an optional prompt.

    This class is based on [`T5ForConditionalGeneration`], extended to deal with images and layout (2D) data.""",
    MARKUSHGRAPHER_START_DOCSTRING,
)
class MarkushgrapherForConditionalGeneration(MarkushgrapherPreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
        "lm_head.weight",
    ]

    def safe_load(self, module, module_states):
        def remove_prefix(state_dict):
            return {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
        return

    def __init__(self, config):
        """
        Note: .from_pretrained() randomly initialize any parameter which is not in the provided states_dict.
        """
        super(MarkushgrapherForConditionalGeneration, self).__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = MarkushgrapherPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MarkushgrapherStack(encoder_config, self.shared, self.patch_embed)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MarkushgrapherStack(decoder_config, self.shared)

        # The weights of the language modeling head are shared with those of the encoder and decoder
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Markushgrapher
        if self.config.architecture_variant == "definition-group-encoder":
            self.definition_groups_dims = {"in": 1024, "out": 1024}
            self.definition_groups_encoder = nn.Sequential(
                nn.Linear(self.definition_groups_dims["in"], self.definition_groups_dims["in"]),
                nn.ReLU(),
                nn.Dropout(p=0.15),
                nn.Linear(self.definition_groups_dims["in"], self.definition_groups_dims["out"]),
            )

        if ("molscribe-encoder" in self.config.architecture_variant) and not (
            ("molscribe-encoder-4" in self.config.architecture_variant)
            or ("molscribe-encoder-6" in self.config.architecture_variant)
        ):
            molscribe_config = SimpleNamespace(
                **{
                    "encoder": "swin_base",
                    "use_checkpoint": True,
                    "formats": ["chartok_coords", "edges"],
                    "vocab_file": os.path.dirname(__file__)
                    + "/../../../../../MolScribe/molscribe/vocab/vocab_chars.json",
                    "coord_bins": 64,
                    "sep_xy": True,
                    "continuous_coords": False,
                    "embed_dim": 256,
                    "enc_pos_emb": False,
                    "decoder_dim": 512,
                    "decoder_layer": 1,
                    "attention_dim": 256,
                    "dec_num_layers": 6,
                    "dec_hidden_size": 256,
                    "dec_attn_heads": 8,
                    "dec_num_queries": 128,
                    "hidden_dropout": 0.1,
                    "attn_dropout": 0.1,
                    "max_relative_positions": 0,
                    "beam_size": 1,
                    "n_best": 1,
                    "predict_coords": False,
                    "save_attns": False,
                    "molblock": False,
                    "compute_confidence": False,
                    "keep_main_molecule": False,
                }
            )
            self.molscribe_encoder = Encoder(molscribe_config, pretrained=False)
            # molscribe_config.encoder_dim = self.molscribe_encoder.n_features
            # self.molscribe_decoder = Decoder(molscribe_config, get_tokenizer(molscribe_config))

        if (self.config.architecture_variant == "molscribe-encoder-5") or (
            self.config.architecture_variant == "molscribe-encoder-7"
        ):  
            self.molscribe_projector = nn.Sequential(
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 1024)
            )

    # Markushgrapher
    def init_molscribe_weights(self):
        if ("molscribe-encoder" in self.config.architecture_variant) and not (
            ("molscribe-encoder-4" in self.config.architecture_variant)
            or ("molscribe-encoder-6" in self.config.architecture_variant)
        ):
            print("Load MolScribe weights")
            states = torch.load(
                os.path.dirname(__file__) + "/../../../../../MolScribe/ckpts/swin_base_char_aux_1m680k.pth",
                map_location=torch.device("cpu"),
            )
            self.safe_load(self.molscribe_encoder, states["encoder"])
            # self.safe_load(self.molscribe_decoder, states['decoder'])

            # Freeze MolScribe encoder
            for param in self.molscribe_encoder.parameters():
                param.requires_grad = False

        if ("molscribe-encoder-4" in self.config.architecture_variant) or (
            "molscribe-encoder-6" in self.config.architecture_variant
        ):
            print("Load MolScribe weights")
            states = torch.load(
                os.path.dirname(__file__) + "/../../../../../MolScribe/ckpts/swin_base_char_aux_1m680k.pth",
                map_location=torch.device("cpu"),
            )
            self.safe_load(self.encoder.molscribe_encoder, states["encoder"])
            # self.safe_load(self.encoder.molscribe_decoder, states['decoder'])

            # Freeze MolScribe encoder
            for param in self.encoder.molscribe_encoder.parameters():
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(MARKUSHGRAPHER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        bbox: Dict[str, Any] = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        definition_groups: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, MarkushgrapherForConditionalGeneration
        >>> from huggingface_hub import hf_hub_download
        >>> from PIL import Image

        >>> # load model and processor
        >>> processor = AutoProcessor.from_pretrained("nielsr/udop-large")
        >>> model = MarkushgrapherForConditionalGeneration.from_pretrained("nielsr/udop-large")

        >>> # load image
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> # inference
        >>> prompt = "Question answering. In which year is the report made?"
        >>> encoding = processor(images=image, text=prompt, return_tensors="pt")
        >>> predicted_ids = model.generate(**encoding)
        >>> print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                bbox=bbox,
                visual_bbox=visual_bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if self.config.architecture_variant == "molscribe-encoder":  # Note: MolScribe weights were not updated
                # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
                pixel_values_resized = F.interpolate(
                    pixel_values, size=(384, 384), mode="bilinear", align_corners=False
                )

                # Get MolScribe features
                molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized)

                # Concatenate the encodings
                # Note: molscribe_features: ([batch_s, 144, 1024]), encoder outputs: ([batch_s, seq_length, 1024]) (seq_length is around 1000)
                # encoder_outputs["attention_mask"] is [1, 1, 1, ..., 1, 0, 0] (of size: [batch_s, seq_length])
                if verbose:
                    print("encoder_outputs", encoder_outputs)
                    print("encoder_outputs[0].size()", encoder_outputs[0].size())
                    print("encoder_outputs[1].size()", encoder_outputs[1].size())
                    print(
                        "torch.cat((encoder_outputs[0], molscribe_features), dim=1).size()",
                        torch.cat((encoder_outputs[0], molscribe_features), dim=1).size(),
                    )
                    print(
                        "torch.cat((encoder_outputs[1], torch.ones(encoder_outputs['attention_mask'].size(0), molscribe_features.size(1)).to(self.device)), dim=1).size()",
                        torch.cat(
                            (
                                encoder_outputs[1],
                                torch.ones(encoder_outputs["attention_mask"].size(0), molscribe_features.size(1)).to(
                                    self.device
                                ),
                            ),
                            dim=1,
                        ).size(),
                    )

                encoder_outputs["last_hidden_state"] = torch.cat((encoder_outputs[0], molscribe_features), dim=1)
                encoder_outputs["attention_mask"] = torch.cat(
                    (
                        encoder_outputs[1],
                        torch.ones(encoder_outputs["attention_mask"].size(0), molscribe_features.size(1)).to(
                            self.device
                        ),
                    ),
                    dim=1,
                )

                if verbose:
                    print("MolScribe parameters:", [p for p in self.molscribe_encoder.parameters()][-1])
                    print("UDOP parameters:", [p for p in self.encoder.parameters()][-1][0])

            if self.config.architecture_variant == "molscribe-encoder-2":  # Note: MolScribe weights were not updated
                if verbose:
                    print("MolScribe encoder parameters")
                    for i, (name, param) in enumerate(self.molscribe_encoder.named_parameters()):
                        if i != 0:
                            continue
                        print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")
                    print("Molscribe eecoder parameters")
                    for i, (name, param) in enumerate(self.molscribe_decoder.named_parameters()):
                        if i != 0:
                            continue
                        print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")

                # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
                pixel_values_resized = torch.stack(
                    [
                        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                        for img in pixel_values
                    ]
                )
                if verbose:
                    print(pixel_values_resized)
                    image_np = pixel_values_resized[0].permute(1, 2, 0).cpu().numpy()
                    plt.imshow(image_np)
                    plt.savefig("test2.png")

                # Get MolScribe features
                molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)

                # Add the encodings to avoid changing dimensions
                encoder_outputs[0][:, : molscribe_features.size(1), :] += molscribe_features
                if verbose:
                    print(self.molscribe_decoder.decode(molscribe_features, hiddens, refs=None))

            if self.config.architecture_variant == "molscribe-encoder-3":
                if verbose:
                    print("MolScribe encoder parameters")
                    for i, (name, param) in enumerate(self.molscribe_encoder.named_parameters()):
                        if i != 0:
                            continue
                        print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")
                    print("Molscribe eecoder parameters")
                    for i, (name, param) in enumerate(self.molscribe_decoder.named_parameters()):
                        if i != 0:
                            continue
                        print(f"Parameter Name: {name}, Shape: {param.shape}, Type: {param.type()}, Value: {param[0]}")

                # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
                pixel_values_resized = torch.stack(
                    [
                        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                        for img in pixel_values
                    ]
                )
                if verbose:
                    print(pixel_values_resized)
                    image_np = pixel_values_resized[0].permute(1, 2, 0).cpu().numpy()
                    plt.imshow(image_np)
                    plt.savefig("test2.png")

                # Get MolScribe features
                molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)

                encoder_outputs["last_hidden_state"] = torch.cat((encoder_outputs[0], molscribe_features), dim=1)
                encoder_outputs["attention_mask"] = torch.cat(
                    (
                        encoder_outputs[1],
                        torch.ones(encoder_outputs["attention_mask"].size(0), molscribe_features.size(1)).to(
                            self.device
                        ),
                    ),
                    dim=1,
                )

            if self.config.architecture_variant == "molscribe-encoder-5":
                # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
                pixel_values_resized = torch.stack(
                    [
                        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                        for img in pixel_values
                    ]
                )

                # Get MolScribe features
                molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)
                molscribe_features_projected = self.molscribe_projector(molscribe_features)

                encoder_outputs["last_hidden_state"] = torch.cat(
                    (molscribe_features_projected, encoder_outputs[0]), dim=1
                )
                encoder_outputs["attention_mask"] = torch.cat(
                    (
                        torch.ones(encoder_outputs["attention_mask"].size(0), molscribe_features_projected.size(1)).to(
                            self.device
                        ),
                        encoder_outputs[1],
                    ),
                    dim=1,
                )

            if self.config.architecture_variant == "molscribe-encoder-7":
                # Resize the image from [batch_s, 3, 512, 512] to [batch_s, 3, 384, 384] for MolScribe
                pixel_values_resized = torch.stack(
                    [
                        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                        for img in pixel_values
                    ]
                )

                # Get MolScribe features
                molscribe_features, hiddens = self.molscribe_encoder(pixel_values_resized, refs=None)
                molscribe_features_projected = self.molscribe_projector(molscribe_features)

                # Normalize molscribe_features_projected to the same range than encoder_outputs[0]
                min_enc_out = encoder_outputs[0].min(dim=1, keepdim=True)[0]
                max_enc_out = encoder_outputs[0].max(dim=1, keepdim=True)[0]
                molscribe_features_projected = (
                    molscribe_features_projected - molscribe_features_projected.min(dim=1, keepdim=True)[0]
                ) / (
                    molscribe_features_projected.max(dim=1, keepdim=True)[0]
                    - molscribe_features_projected.min(dim=1, keepdim=True)[0]
                )
                molscribe_features_projected = molscribe_features_projected * (max_enc_out - min_enc_out) + min_enc_out

                # Add separation token
                separation_token = torch.full(
                    (encoder_outputs["last_hidden_state"].size(0), 1, encoder_outputs["last_hidden_state"].size(2)),
                    fill_value=0,
                ).to(self.device)

                # Create a corresponding attention mask for the separation token
                separation_token_mask = torch.ones((encoder_outputs["attention_mask"].size(0), 1)).to(self.device)

                # Concatenate
                encoder_outputs["last_hidden_state"] = torch.cat(
                    (molscribe_features_projected, separation_token, encoder_outputs[0]), dim=1
                )
                encoder_outputs["attention_mask"] = torch.cat(
                    (
                        torch.ones(encoder_outputs["attention_mask"].size(0), molscribe_features_projected.size(1)).to(
                            self.device
                        ),
                        separation_token_mask,
                        encoder_outputs[1],
                    ),
                    dim=1,
                )
                if verbose:
                    print("encoder_outputs size", encoder_outputs[0].size())

            if self.config.architecture_variant == "definition-group-encoder":
                if verbose:
                    print("definition_groups:", definition_groups)
                    print(f"encoder_outputs size:{encoder_outputs[0].size()}")
                    print("bbox size:", bbox.size())
                    print("inputs_ids size:", input_ids.size())
                    # (batch_size, sequence_length, hidden_size)
                    # Note: indices in definition_groups are the same than encodings indices in encoder_outputs

                # Create "definition_groups_input" by replacing by zeros the indices in "encoder_outputs" which are not part of each definition group
                nb_groups = definition_groups.size(1)
                definition_groups_input = torch.zeros(
                    encoder_outputs[0].size(0), nb_groups, encoder_outputs[0].size(1), 1024
                ).to(self.device)
                if verbose:
                    print("definition_groups_input size:", definition_groups_input.size())
                for batch_idx in range(encoder_outputs[0].size(0)):
                    for group_idx in range(nb_groups):
                        indices = [
                            definition_groups[batch_idx, group_idx][0],
                            definition_groups[batch_idx, group_idx][1],
                        ]
                        indices += list(
                            range(
                                definition_groups[batch_idx, group_idx][2],
                                definition_groups[batch_idx, group_idx][3] + 1,
                            )
                        )  # Should the last index be included?.
                        if indices == [-1, -1, -1]:
                            continue
                        indices = torch.tensor(indices).to(self.device)
                        definition_groups_input[batch_idx, :, indices, :] += encoder_outputs[0][batch_idx, indices, :]

                        if verbose:
                            print("indices:", indices)
                            print(
                                "encoder_outputs[0][batch_idx, indices.long()] size:",
                                encoder_outputs[0][batch_idx, indices[0] : indices[1]].size(),
                            )

                if verbose:
                    print("definition_groups_input size:", definition_groups_input.size())
                # Apply linear layer
                # (batch_size, nb_groups, sequence_length, 1024)
                definition_groups_encoding = self.definition_groups_encoder(definition_groups_input)
                if verbose:
                    print("definition_groups_encoding size:", definition_groups_encoding.size())

                # Reduce "sequence_length" with mean
                # (batch_size, nb_groups, 1024)
                definition_groups_encoding = definition_groups_encoding.mean(dim=2)

                # Concatenate with encoder_outputs (batch_size, sequence_length, 1024)
                # (batch_size, nb_groups + sequence_length, 1024)
                # For first testing, add to first dimensions to keep a size of (batch_size, sequence_length, 1024)
                if verbose:
                    print("definition_groups_encoding size:", definition_groups_encoding.size())
                encoder_outputs[0][:, :nb_groups, :] += definition_groups_encoding

        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[2:] + (encoder_outputs[0],) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "bbox": kwargs.get("bbox", None),
            "pixel_values": kwargs.get("pixel_values", None),
            "visual_bbox": kwargs.get("visual_bbox", None),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare UDOP Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    MARKUSHGRAPHER_START_DOCSTRING,
)
class MarkushgrapherEncoderModel(MarkushgrapherPreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    def __init__(self, config: MarkushgrapherConfig):
        super().__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = MarkushgrapherPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MarkushgrapherStack(encoder_config, self.shared, self.patch_embed)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MARKUSHGRAPHER_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithAttentionMask, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        bbox: Dict[str, Any] = None,
        attention_mask: Tensor = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithAttentionMask]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MarkushgrapherEncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("nielsr/udop-large")
        >>> model = MarkushgrapherEncoderModel.from_pretrained("nielsr/udop-large")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            visual_bbox=visual_bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
