python -u train.py \
       --train-batch-size 4 \
       --eval-batch-size 1 \
       --gpu 0 \
       --lr 0.0001 \
       --c_feat 16 \
       --epochs 10000 \
       --resume_epoch 0 \
       --patience 10 \
       --loss-delta 0 \
       --es-mode "min"
