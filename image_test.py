import argparse
import os

import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from torchvision.transforms import  RandomCrop, ToTensor, ToPILImage

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder
from model import Generator
from data_utils import display_transform


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size 
UPSCALE_FACTOR = opt.upscale_factor 
NUM_EPOCHS = opt.num_epochs 

val_set = TestDatasetFromFolder('D:/Dataset/', upscale_factor=UPSCALE_FACTOR) 

MODEL_NAME = 'netG_epoch_1_300.pth'
netG = Generator(UPSCALE_FACTOR).eval()
netG.cuda()
netG.load_state_dict(torch.load('D:/Project/NewProject/ResNetFusion-master/model1/' + MODEL_NAME))
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

epoch =1
out_path = 'D:/Dataset/results_' + str(UPSCALE_FACTOR) + '/'


val_bar = tqdm(val_loader) 
val_images = []

for val_lr , val_lr_restore, val_hr in val_bar:
    batch_size = val_lr.size(0)
    lr = Variable(val_lr)
    hr = Variable(val_hr)
    if torch.cuda.is_available():
        lr = lr.cuda()
        hr = hr.cuda()

    sr = netG(lr)

    val_images.extend(
            [display_transform()(val_lr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
val_images = torch.stack(val_images)

val_images = torch.split(val_images, 1 ,dim=0)
start = time.time()
for i,image in enumerate(val_images):
    print('{}th size {}'.format(i,image.size()))
val_save_bar = tqdm(val_images, desc='[saving training results]')
index = 1
for image in val_save_bar:
    image = utils.make_grid(image, nrow=3, padding=2,scale_each=True)
    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), nrow=3,padding=2)#验证集存储数据
    end = time.time()
    print(end-start)
    index += 1
