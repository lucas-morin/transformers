# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_markushgrapher": ["MARKUSHGRAPHER_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarkushgrapherConfig"],
    "processing_markushgrapher": ["MarkushgrapherProcessor"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_markushgrapher"] = ["MarkushgrapherTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_markushgrapher_fast"] = ["MarkushgrapherTokenizerFast"]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_markushgrapher"] = ["MarkushgrapherImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_markushgrapher"] = [
        "MARKUSHGRAPHER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarkushgrapherForConditionalGeneration",
        "MarkushgrapherPreTrainedModel",
        "MarkushgrapherModel",
        "MarkushgrapherEncoderModel",
    ]

if TYPE_CHECKING:
    from .configuration_markushgrapher import MARKUSHGRAPHER_PRETRAINED_CONFIG_ARCHIVE_MAP, MarkushgrapherConfig
    from .processing_markushgrapher import MarkushgrapherProcessor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_markushgrapher import MarkushgrapherTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_markushgrapher_fast import MarkushgrapherTokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_markushgrapher import MarkushgrapherImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_markushgrapher import (
            MARKUSHGRAPHER_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarkushgrapherEncoderModel,
            MarkushgrapherForConditionalGeneration,
            MarkushgrapherModel,
            MarkushgrapherPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
