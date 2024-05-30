
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random

from torch.utils.data import DataLoader
from resnet import  wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn

from test import evaluation
from torch.nn import functional as F
import warnings
from SE import SELayer

warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):

        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                   b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):

    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):

        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)


    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    print(_class_)
    epochs = 120
    learning_rate = 0.005
    batch_size = 8
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '' + _class_ + ''
    test_path = '' + _class_
    ckp_path = '' + ''+_class_+''
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    se1 = SELayer(256, reduction_ratio=8).to(device)
    se2 = SELayer(512, reduction_ratio=8).to(device)
    se3 = SELayer(1024, reduction_ratio=8).to(device)
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)

            inputs1 = []  # 1
            outputs1 = []  # 2
            inputs = encoder(img)

            outputs = decoder(bn(inputs))

            inputs1.append(se1(inputs[0]))  # 1
            outputs1.append(se1(outputs[0]))  # 1
            inputs1.append(se2(inputs[1]))  # 1
            outputs1.append(se2(outputs[1]))  # 1
            inputs1.append(se3(inputs[2]))  # 1
            outputs1.append(se3(outputs[2]))  # 1

            loss = loss_fucntion(inputs1, outputs1)  # 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 1 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device, se1, se2, se3)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px



