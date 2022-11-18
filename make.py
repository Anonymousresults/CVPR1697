import torch
from collections import OrderedDict

# layer4.0.downsample.1.weight

new = OrderedDict()

new['model'] = OrderedDict()

for k,v in torch.load('checkpoint.pth', map_location='cpu')['model'].items():
    if 'norm_q' in k:
        print(k,v)
    new['model'][k] = v

torch.save(new, '2bit_qdetr.pth')
