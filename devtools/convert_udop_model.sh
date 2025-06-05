#!/bin/bash

. ./venv/bin/activate
pip install pytesseract

python src/transformers/models/udop/convert_udop_to_pytorch.py \
    --pytorch_dump_folder /Users/nikoslivathinos/code/nli/udop/udop_hf_converted_models/
