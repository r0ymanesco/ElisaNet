import os
import network

import torch
import skimage.io as io

from util import np_to_torch
from test_options import parser


args = parser.parse_args()
fns = args.images.split(',')
print('Images to be tested:')
print(fns)

elisa_net = network.ElisaNet(c_feat=16)
elisa_net.load_state_dict(torch.load('ELISA-NET_WEIGHTS.pth'))

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    elisa_net = elisa_net.cuda()

img_tensor = []
for img_fn in fns:
    img = io.imread(img_fn)
    if img.shape != (3648, 2736, 3):
        AssertionError('Image shape does not conform to resolution 2736x3648 RGB.')
    else:
        img = img[456:-456, :, :]
        img = np_to_torch(img) / 255.
        if args.gpu is not None:
            img = img.cuda()
        img_tensor.append(img)

img_tensor = torch.stack(img_tensor, dim=0)
with torch.no_grad():
    predictions = torch.round(
        torch.sigmoid(elisa_net(img_tensor))
    )

print('Results:')
for img_idx, img_fn in enumerate(fns):
    if predictions[img_idx].item() == 0:
        print('{}: Positive'.format(img_fn))
    else:
        print('{}: Negative'.format(img_fn))
