Requirements:
pytorch 1.7.1
torchvision 0.8.2
cudatoolkit 11.0

Test 4-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch_size 4 --resume 4bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --eval --quant --n_bit 4

Test 3-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch_size 4 --resume 3bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --eval --quant --n_bit 3

Test 2-bit Q-DETR on VOC:

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch_size 4 --resume 2bit_qdetr.pth --dataset_file voc --coco_path /home/data/voc/VOCdevkit/ --backbone resnet50 --eval --quant --n_bit 2

The test results can be replicated with 4 GPUs and mini batch size set as 4. We recommend this test setting.

The checkpoints can be fetched in https://drive.google.com/drive/folders/1mg6ynNpm3r3UOP5OLFR9E5PDuyag0BB3?usp=sharing.
