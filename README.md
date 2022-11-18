Requirements:
pytorch 1.7.1
torchvision 0.8.2
cudatoolkit 11.0

Test 4-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch_size 2 --resume 4bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --epochs 50 --lr_drop 35 --dropout 0.0 --eval --quant --n_bit 4

Test 3-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch_size 2 --resume 3bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --epochs 50 --lr_drop 35 --dropout 0.0 --eval --quant --n_bit 3

Test 2-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --batch_size 2 --resume 2bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --epochs 50 --lr_drop 35 --dropout 0.0 --eval --quant --n_bit 2

The checkpoints can be fetched in 
