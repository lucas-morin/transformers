# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BEiT model."""


import inspect
import unittest

from datasets import load_dataset
from packaging import version

from transformers import BeitConfig
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_torch_multi_gpu, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        MODEL_MAPPING,
        BeitForImageClassification,
        BeitForMaskedImageModeling,
        BeitForSemanticSegmentation,
        BeitModel,
    )
    from transformers.models.beit.modeling_beit import BEIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    import PIL
    from PIL import Image

    from transformers import BeitImageProcessor


class BeitModelTester:
    def __init__(
        self,
        parent,
        vocab_size=100,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        out_indices=[0, 1, 2, 3],
    ):
        self.parent = parent
        self.vocab_size = 100
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.out_indices = out_indices
        self.num_labels = num_labels

        # in BeiT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return BeitConfig(
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = BeitModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(self, config, pixel_values, labels, pixel_labels):
        model = BeitForMaskedImageModeling(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length - 1, self.vocab_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.type_sequence_label_size
        model = BeitForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = BeitForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = BeitForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class BeitModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as BEiT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (BeitModel, BeitForImageClassification, BeitForMaskedImageModeling, BeitForSemanticSegmentation)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BeitModel,
            "image-classification": BeitForImageClassification,
            "image-segmentation": BeitForSemanticSegmentation,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = BeitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BeitConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="BEiT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(reason="BEiT has some layers using `add_module` which doesn't work well with `nn.DataParallel`")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            # we don't test BeitForMaskedImageModeling
            if model_class in [*get_values(MODEL_MAPPING), BeitForMaskedImageModeling]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.model_tester.is_training:
            return

        config.use_cache = False
        config.return_dict = True

        for model_class in self.all_model_classes:
            # we don't test BeitForMaskedImageModeling
            if (
                model_class in [*get_values(MODEL_MAPPING), BeitForMaskedImageModeling]
                or not model_class.supports_gradient_checkpointing
            ):
                continue

            model = model_class(config)
            model.gradient_checkpointing_enable()
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # we skip lambda parameters as these require special initial values
                # determined by config.layer_scale_init_value
                if "lambda" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @slow
    def test_model_from_pretrained(self):
        for model_name in BEIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BeitModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class BeitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_inference_masked_image_modeling_head(self):
        model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # prepare bool_masked_pos
        bool_masked_pos = torch.ones((1, 196), dtype=torch.bool).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 196, 8192))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-3.2437, 0.5072, -13.9174], [-3.2456, 0.4948, -13.9401], [-3.2033, 0.5121, -13.8550]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(logits[bool_masked_pos][:3, :3], expected_slice, atol=1e-2))

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.2385, -1.0987, -1.0108]).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 281
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)

    @slow
    def test_inference_image_classification_head_imagenet_22k(self):
        model = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224-pt22k-ft22k").to(
            torch_device
        )

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 21841))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([1.6881, -0.2787, 0.5901]).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 2396
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)

    @slow
    def test_inference_semantic_segmentation(self):
        model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        model = model.to(torch_device)

        image_processor = BeitImageProcessor(do_resize=True, size=640, do_center_crop=False)

        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 150, 160, 160))
        self.assertEqual(logits.shape, expected_shape)

        is_pillow_less_than_9 = version.parse(PIL.__version__) < version.parse("9.0.0")

        if is_pillow_less_than_9:
            expected_slice = torch.tensor(
                [
                    [[-4.9225, -2.3954, -3.0522], [-2.8822, -1.0046, -1.7561], [-2.9549, -1.3228, -2.1347]],
                    [[-5.8168, -3.4129, -4.0778], [-3.8651, -2.2214, -3.0277], [-3.8356, -2.4643, -3.3535]],
                    [[-0.0078, 3.9952, 4.0754], [2.9856, 4.6944, 5.0035], [3.2413, 4.7813, 4.9969]],
                ],
                device=torch_device,
            )
        else:
            expected_slice = torch.tensor(
                [
                    [[-4.8960, -2.3688, -3.0355], [-2.8478, -0.9836, -1.7418], [-2.9449, -1.3332, -2.1456]],
                    [[-5.8081, -3.4124, -4.1006], [-3.8561, -2.2081, -3.0323], [-3.8365, -2.4601, -3.3669]],
                    [[-0.0309, 3.9868, 4.0540], [2.9640, 4.6877, 4.9976], [3.2081, 4.7690, 4.9942]],
                ],
                device=torch_device,
            )

        self.assertTrue(torch.allclose(logits[0, :3, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_post_processing_semantic_segmentation(self):
        model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        model = model.to(torch_device)

        image_processor = BeitImageProcessor(do_resize=True, size=640, do_center_crop=False)

        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        outputs.logits = outputs.logits.detach().cpu()

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(500, 300)])
        expected_shape = torch.Size((500, 300))
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = torch.Size((160, 160))
        self.assertEqual(segmentation[0].shape, expected_shape)
