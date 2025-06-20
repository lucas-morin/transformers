# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MaskFormer model."""

import copy
import inspect
import unittest

import numpy as np

from tests.test_modeling_common import floats_tensor
from transformers import DetrConfig, MaskFormerConfig, SwinConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import MaskFormerForInstanceSegmentation, MaskFormerModel

    if is_vision_available():
        from transformers import MaskFormerImageProcessor

if is_vision_available():
    from PIL import Image


class MaskFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        use_auxiliary_loss=False,
        num_queries=10,
        num_channels=3,
        min_size=32 * 4,
        max_size=32 * 6,
        num_labels=4,
        mask_feature_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.mask_feature_size = mask_feature_size
        # This is passed to the decoder config. We add it to the model tester here for testing
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size]).to(
            torch_device
        )

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.min_size, self.max_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, pixel_mask, mask_labels, class_labels

    def get_config(self):
        return MaskFormerConfig.from_backbone_and_decoder_configs(
            backbone_config=SwinConfig(
                depths=[1, 1, 1, 1],
                embed_dim=16,
                hidden_size=32,
                num_heads=[1, 1, 2, 2],
            ),
            decoder_config=DetrConfig(
                decoder_ffn_dim=64,
                decoder_layers=self.num_hidden_layers,
                decoder_attention_heads=self.num_attention_heads,
                encoder_ffn_dim=64,
                encoder_layers=self.num_hidden_layers,
                encoder_attention_heads=self.num_attention_heads,
                num_queries=self.num_queries,
                d_model=self.mask_feature_size,
            ),
            mask_feature_size=self.mask_feature_size,
            fpn_feature_size=self.mask_feature_size,
            num_channels=self.num_channels,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def check_output_hidden_state(self, output, config):
        encoder_hidden_states = output.encoder_hidden_states
        pixel_decoder_hidden_states = output.pixel_decoder_hidden_states
        transformer_decoder_hidden_states = output.transformer_decoder_hidden_states

        self.parent.assertTrue(len(encoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(pixel_decoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(transformer_decoder_hidden_states), config.decoder_config.decoder_layers)

    def create_and_check_maskformer_model(self, config, pixel_values, pixel_mask, output_hidden_states=False):
        with torch.no_grad():
            model = MaskFormerModel(config=config)
            model.to(torch_device)
            model.eval()

            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            output = model(pixel_values, output_hidden_states=True)
        # the correct shape of output.transformer_decoder_hidden_states ensure the correcteness of the
        # encoder and pixel decoder
        self.parent.assertEqual(
            output.transformer_decoder_last_hidden_state.shape,
            (self.batch_size, self.num_queries, self.mask_feature_size),
        )
        # let's ensure the other two hidden state exists
        self.parent.assertTrue(output.pixel_decoder_last_hidden_state is not None)
        self.parent.assertTrue(output.encoder_last_hidden_state is not None)

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_maskformer_instance_segmentation_head_model(
        self, config, pixel_values, pixel_mask, mask_labels, class_labels
    ):
        model = MaskFormerForInstanceSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # let's still check that all the required stuff is there
            self.parent.assertTrue(result.transformer_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.pixel_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.encoder_last_hidden_state is not None)
            # okay, now we need to check the logits shape
            # due to the encoder compression, masks have a //4 spatial size
            self.parent.assertEqual(
                result.masks_queries_logits.shape,
                (self.batch_size, self.num_queries, self.min_size // 4, self.max_size // 4),
            )
            # + 1 for null class
            self.parent.assertEqual(
                result.class_queries_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1)
            )

        with torch.no_grad():
            result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            result = model(pixel_values)

            comm_check_on_output(result)

            result = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels
            )

        comm_check_on_output(result)

        self.parent.assertTrue(result.loss is not None)
        self.parent.assertEqual(result.loss.shape, torch.Size([1]))


@require_torch
class MaskFormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MaskFormerModel, MaskFormerForInstanceSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": MaskFormerModel, "image-segmentation": MaskFormerForInstanceSegmentation}
        if is_torch_available()
        else {}
    )

    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = MaskFormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MaskFormerConfig, has_text_modality=False)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class in [MaskFormerForInstanceSegmentation]:
                inputs_dict["mask_labels"] = torch.zeros(
                    (
                        self.model_tester.batch_size,
                        self.model_tester.num_labels,
                        self.model_tester.min_size,
                        self.model_tester.max_size,
                    ),
                    dtype=torch.float32,
                    device=torch_device,
                )
                inputs_dict["class_labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.num_labels), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_maskformer_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_maskformer_model(config, **inputs, output_hidden_states=False)

    def test_maskformer_instance_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_maskformer_instance_segmentation_head_model(*config_and_inputs)

    @unittest.skip(reason="MaskFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MaskFormer does not have a get_input_embeddings method")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="MaskFormer is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="MaskFormer does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(
        reason="MaskFormer has some layers using `add_module` which doesn't work well with `nn.DataParallel`"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["facebook/maskformer-swin-small-coco"]:
            model = MaskFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_with_labels(self):
        size = (self.model_tester.min_size,) * 2
        inputs = {
            "pixel_values": torch.randn((2, 3, *size), device=torch_device),
            "mask_labels": torch.randn((2, 10, *size), device=torch_device),
            "class_labels": torch.zeros(2, 10, device=torch_device).long(),
        }

        model = MaskFormerForInstanceSegmentation(MaskFormerConfig()).to(torch_device)
        outputs = model(**inputs)
        self.assertTrue(outputs.loss is not None)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_maskformer_model(config, **inputs, output_hidden_states=True)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # Check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            # encoder_hidden_states, pixel_decoder_hidden_states, transformer_decoder_hidden_states, hidden_states
            added_hidden_states = 4
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

    def test_retain_grad_hidden_states_attentions(self):
        # only MaskFormerForInstanceSegmentation has the loss
        model_class = self.all_model_classes[1]
        config, pixel_values, pixel_mask, mask_labels, class_labels = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        config.output_attentions = True

        model = model_class(config)
        model.to(torch_device)
        model.train()

        outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states[0]
        pixel_decoder_hidden_states.retain_grad()
        # we requires_grad=True in inputs_embeds (line 2152), the original implementation don't
        transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states[0]
        transformer_decoder_hidden_states.retain_grad()

        attentions = outputs.attentions[0]
        attentions.retain_grad()

        outputs.loss.backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(pixel_decoder_hidden_states.grad)
        self.assertIsNotNone(transformer_decoder_hidden_states.grad)
        self.assertIsNotNone(attentions.grad)


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@slow
class MaskFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-small-coco")
            if is_vision_available()
            else None
        )

    def test_inference_no_head(self):
        model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-small-coco").to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 800, 1088))

        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice_hidden_state = torch.tensor(
            [[-0.0482, 0.9228, 0.4951], [-0.2547, 0.8017, 0.8527], [-0.0069, 0.3385, -0.0089]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[-0.8422, -0.8434, -0.9718], [-1.0144, -0.5565, -0.4195], [-1.0038, -0.4484, -0.1961]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[0.2852, -0.0159, 0.9735], [0.6254, 0.1858, 0.8529], [-0.0680, -0.4116, 1.8413]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_last_hidden_state[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

    def test_inference_instance_segmentation_head(self):
        model = (
            MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")
            .to(torch_device)
            .eval()
        )
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 800, 1088))

        with torch.no_grad():
            outputs = model(**inputs)
        # masks_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        self.assertEqual(
            masks_queries_logits.shape,
            (1, model.config.decoder_config.num_queries, inputs_shape[-2] // 4, inputs_shape[-1] // 4),
        )
        expected_slice = [
            [-1.3737124, -1.7724937, -1.9364233],
            [-1.5977281, -1.9867939, -2.1523695],
            [-1.5795398, -1.9269832, -2.093942],
        ]
        expected_slice = torch.tensor(expected_slice).to(torch_device)
        self.assertTrue(torch.allclose(masks_queries_logits[0, 0, :3, :3], expected_slice, atol=TOLERANCE))
        # class_queries_logits
        class_queries_logits = outputs.class_queries_logits
        self.assertEqual(
            class_queries_logits.shape, (1, model.config.decoder_config.num_queries, model.config.num_labels + 1)
        )
        expected_slice = torch.tensor(
            [
                [1.6512e00, -5.2572e00, -3.3519e00],
                [3.6169e-02, -5.9025e00, -2.9313e00],
                [1.0766e-04, -7.7630e00, -5.1263e00],
            ]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_instance_segmentation_head_resnet_backbone(self):
        model = (
            MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-resnet101-coco-stuff")
            .to(torch_device)
            .eval()
        )
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 800, 1088))

        with torch.no_grad():
            outputs = model(**inputs)
        # masks_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        self.assertEqual(
            masks_queries_logits.shape,
            (1, model.config.decoder_config.num_queries, inputs_shape[-2] // 4, inputs_shape[-1] // 4),
        )
        expected_slice = [[-0.9046, -2.6366, -4.6062], [-3.4179, -5.7890, -8.8057], [-4.9179, -7.6560, -10.7711]]
        expected_slice = torch.tensor(expected_slice).to(torch_device)
        self.assertTrue(torch.allclose(masks_queries_logits[0, 0, :3, :3], expected_slice, atol=TOLERANCE))
        # class_queries_logits
        class_queries_logits = outputs.class_queries_logits
        self.assertEqual(
            class_queries_logits.shape, (1, model.config.decoder_config.num_queries, model.config.num_labels + 1)
        )
        expected_slice = torch.tensor(
            [[4.7188, -3.2585, -2.8857], [6.6871, -2.9181, -1.2487], [7.2449, -2.2764, -2.1874]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_slice, atol=TOLERANCE))

    @require_torch_gpu
    def test_inference_fp16(self):
        model = (
            MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-resnet101-coco-stuff")
            .to(torch_device, dtype=torch.float16)
            .eval()
        )
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        with torch.no_grad():
            _ = model(**inputs)

    def test_with_segmentation_maps_and_loss(self):
        model = (
            MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")
            .to(torch_device)
            .eval()
        )
        image_processor = self.default_image_processor

        inputs = image_processor(
            [np.zeros((3, 400, 333)), np.zeros((3, 400, 333))],
            segmentation_maps=[np.zeros((384, 384)).astype(np.float32), np.zeros((384, 384)).astype(np.float32)],
            return_tensors="pt",
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(torch_device)
        inputs["mask_labels"] = [el.to(torch_device) for el in inputs["mask_labels"]]
        inputs["class_labels"] = [el.to(torch_device) for el in inputs["class_labels"]]

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(outputs.loss is not None)
