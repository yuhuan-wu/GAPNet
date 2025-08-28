

PYTHONPATH=$(pwd):$PYTHONPATH python3 scripts/train.py --max_epochs 30 \
                                        --num_workers 6 \
                                        --batch_size 32 \
                                        --lr_mode poly \
                                        --lr 1.7e-4 \
                                        --width 384 \
                                        --height 384 \
                                        --iter_size 1 \
                                        --arch mobilenetv2 \
                                        --ms 1 \
                                        --ms1 0 \
                                        --bcedice 1 \
                                        --adam_beta2 0.99 \
                                        --group_lr 0 \
                                        --gpu_id 0 \
                                        --seed 2023 \
                                        --savedir ./snaps/gapnet/
        
    