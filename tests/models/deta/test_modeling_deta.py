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
"""Testing suite for the PyTorch DETA model."""


import inspect
import math
import unittest

from transformers import DetaConfig, ResNetConfig, is_torch_available, is_torchvision_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torchvision, require_vision, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import DetaForObjectDetection, DetaModel


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class DetaModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=12,
        num_channels=3,
        image_size=196,
        n_targets=8,
        num_labels=91,
        num_feature_levels=4,
        encoder_n_points=2,
        decoder_n_points=6,
        two_stage=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.image_size = image_size
        self.n_targets = n_targets
        self.num_labels = num_labels
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length = (
            math.ceil(self.image_size / 8) ** 2
            + math.ceil(self.image_size / 16) ** 2
            + math.ceil(self.image_size / 32) ** 2
            + math.ceil(self.image_size / 64) ** 2
        )
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size], device=torch_device)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                target["masks"] = torch.rand(self.n_targets, self.image_size, self.image_size, device=torch_device)
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=10,
            hidden_sizes=[10, 20, 30, 40],
            depths=[1, 1, 2, 1],
            hidden_act="relu",
            num_labels=3,
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        return DetaConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            num_feature_levels=self.num_feature_levels,
            encoder_n_points=self.encoder_n_points,
            decoder_n_points=self.decoder_n_points,
            two_stage=self.two_stage,
            backbone_config=resnet_config,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def create_and_check_deta_model(self, config, pixel_values, pixel_mask, labels):
        model = DetaModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_queries, self.hidden_size))

    def create_and_check_deta_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = DetaForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_torchvision
class DetaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DetaModel, DetaForObjectDetection) if is_torchvision_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": DetaModel, "object-detection": DetaForObjectDetection}
        if is_torchvision_available()
        else {}
    )
    is_encoder_decoder = True
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name == "ObjectDetectionPipelineTests":
            return True

        return False

    @unittest.skip("Skip for now. PR #22437 causes some loading issue. See (not merged) #22656 for some discussions.")
    def test_can_use_safetensors(self):
        super().test_can_use_safetensors()

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "DetaForObjectDetection":
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    target["masks"] = torch.ones(
                        self.model_tester.n_targets,
                        self.model_tester.image_size,
                        self.model_tester.image_size,
                        device=torch_device,
                        dtype=torch.float,
                    )
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = DetaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DetaConfig, has_text_modality=False)

    def test_config(self):
        # we don't test common_properties and arguments_init as these don't apply for DETA
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_deta_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_deta_model(*config_and_inputs)

    def test_deta_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_deta_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="DETA does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="DETA does not have a get_input_embeddings method")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="DETA is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="DETA does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

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
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.encoder_n_points,
                ],
            )
            out_len = len(outputs)

            correct_outlen = 8

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            # Object Detection model returns pred_logits and pred_boxes
            if model_class.__name__ == "DetaForObjectDetection":
                correct_outlen += 2

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, self.model_tester.num_queries, self.model_tester.num_queries],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.decoder_n_points,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.encoder_n_points,
                ],
            )

    # removed retain_grad and grad on decoder_hidden_states, as queries don't require grad
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        # we take the second output since last_hidden_state is the second item
        output = outputs[1]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        decoder_attentions = outputs.decoder_attentions[0]
        decoder_attentions.retain_grad()

        cross_attentions = outputs.cross_attentions[0]
        cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(decoder_attentions.grad)
        self.assertIsNotNone(cross_attentions.grad)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = ["pixel_values", "pixel_mask"]
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask", "encoder_outputs"]
                    if "head_mask" and "decoder_head_mask" in arg_names
                    else []
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["pixel_values", "pixel_mask"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip(reason="Model doesn't use tied weights")
    def test_tied_model_weights_key_ignore(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            # Skip the check for the backbone
            for name, module in model.named_modules():
                if module.__class__.__name__ == "DetaBackboneWithPositionalEncodings":
                    backbone_params = [f"{name}.{key}" for key in module.state_dict().keys()]
                    break

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if (
                        "level_embed" in name
                        or "sampling_offsets.bias" in name
                        or "value_proj" in name
                        or "output_proj" in name
                        or "reference_points" in name
                        or name in backbone_params
                    ):
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torchvision
@require_vision
@slow
class DetaModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("jozhang97/deta-resnet-50") if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = DetaForObjectDetection.from_pretrained("jozhang97/deta-resnet-50").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape_logits = torch.Size((1, 300, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_logits = torch.tensor(
            [[-7.3978, -2.5406, -4.1668], [-8.2684, -3.9933, -3.8096], [-7.0515, -3.7973, -5.8516]]
        ).to(torch_device)
        expected_boxes = torch.tensor(
            [[0.5043, 0.4973, 0.9998], [0.2542, 0.5489, 0.4748], [0.5490, 0.2765, 0.0570]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, 300, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4))

        # verify postprocessing
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.6392, 0.6276, 0.5546, 0.5260, 0.4706], device=torch_device)
        expected_labels = [75, 17, 17, 75, 63]
        expected_slice_boxes = torch.tensor([40.5866, 73.2107, 176.1421, 117.1751], device=torch_device)

        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-4))
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes))

    def test_inference_object_detection_head_swin_backbone(self):
        model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape_logits = torch.Size((1, 300, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_logits = torch.tensor(
            [[-7.6308, -2.8485, -5.3737], [-7.2037, -4.5505, -4.8027], [-7.2943, -4.2611, -4.6617]]
        ).to(torch_device)
        expected_boxes = torch.tensor(
            [[0.4987, 0.4969, 0.9999], [0.2549, 0.5498, 0.4805], [0.5498, 0.2757, 0.0569]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, 300, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4))

        # verify postprocessing
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.6831, 0.6826, 0.5684, 0.5464, 0.4392], device=torch_device)
        expected_labels = [17, 17, 75, 75, 63]
        expected_slice_boxes = torch.tensor([345.8478, 23.6754, 639.8562, 372.8265], device=torch_device)

        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-4))
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes))
