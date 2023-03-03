#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install -r requirements.txt

FILE=${DATA_DIR}/movenet_lightning.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite' \
    -o ${FILE}
fi