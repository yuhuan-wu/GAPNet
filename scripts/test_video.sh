#!/bin/bash

# Test script for two-stage model with multi-modality input
declare -a supervisions=("8")
declare -a datasets=("DAVIS_test" "SegTrack-V2_test" "ViSal_test" "FBMS_test")

# Configuration
SAVE_DIR="./test_results"
METHOD_TAG="gapnet_video"
MODEL_PATH="pretrained/gapnet_video.pth"


for dataset in "${datasets[@]}"
    do
        echo "Testing on dataset: ${dataset}"
        PYTHONPATH=$(pwd):$PYTHONPATH python3 scripts/test_video.py \
        --pretrained $MODEL_PATH \
        --arch mobilenetv2 \
        --dds 1 \
        --gbg 1 \
        --kvc 0 \
        --qc 1 \
        --low_global_vit 0 \
        --vit_dwconv 1 \
        --dilation_opt 1 \
        --dataset_name $dataset \
        --save_dir $SAVE_DIR \
        --method_tag $METHOD_TAG
    done
done