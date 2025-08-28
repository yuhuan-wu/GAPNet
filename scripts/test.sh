#!/bin/bash


PYTHONPATH=$(pwd):$PYTHONPATH  python3 scripts/test.py \
        --pretrained ./pretrained/gapnet.pth \
        --arch mobilenetv2 


