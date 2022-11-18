python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--batch_size 2 --resume 2bit_qdetr.pth \
--dataset_file voc --coco_path /home/data/voc/VOCdevkit/ \
--backbone resnet50 \
--epochs 50 --lr_drop 35 --dropout 0.0 --eval --quant --n_bit 2

