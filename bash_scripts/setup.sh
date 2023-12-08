#!/bin/bash

python -m spacy download en_core_web_sm
mkdir data/datasets
mkdir data/datasets/quality
curl -o data/datasets/quality/QuALITY.v1.0.1.htmlstripped.train https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train
curl -o data/datasets/quality/QuALITY.v1.0.1.htmlstripped.dev https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
curl -o data/datasets/quality/QuALITY.v1.0.1.htmlstripped.test https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test