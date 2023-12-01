#!/bin/bash

pip install bitsandbytes;
pip install --pre -v torch --index-url https://download.pytorch.org/whl/nightly/cu121 --prefix=/ext3/miniconda3;
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --force-reinstall --upgrade flash-attn --no-build-isolation;