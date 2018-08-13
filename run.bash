#!/bin/bash

# Start docker container
#
# Usage:
#     sudo ./runRos.sh

nvidia-docker run -it \
                  --rm \
                  -v ${PWD}/models:/opt/caffe/models \
                  -v ${PWD}/crnn/model:/opt/caffe/crnn/model \
                  tbpp_crnn:gpu bash
