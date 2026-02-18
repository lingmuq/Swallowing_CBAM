import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet_non_oce import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset1 import SwallowingDataset
import torch.backends.cudnn as cudnn
import argparse
from test2 import evaluation, visualization, test
from torch.nn import functional as F


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
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # 确保保存路径存在
    os.makedirs('./checkpoints_non_OCE', exist_ok=True)
    os.makedirs('./train_logs', exist_ok=True)  # <--- 新建日志保存的文件夹

    # 设置日志文件路径
    log_file_path = f'./train_logs/{_class_}_train_log.txt'
    log_file = open(log_file_path, 'w')  # <--- 打开txt文件，准备记录

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = './1111/' + _class_ + '/train'
    test_path = './1111/' + _class_
    ckp_path = './checkpoints_non_OCE/' + 'wres50_' + _class_ + '.pth'

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SwallowingDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+ list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []

        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            # tmp = bn(inputs)
            tmp = inputs[2] 
            outputs = decoder(tmp)
            loss = loss_fucntion(inputs, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        log_file.write(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {np.mean(loss_list):.4f}\n')  # <--- 记录每轮loss
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px, best_th, best_f1 = evaluation(encoder, bn, decoder, test_dataloader, device)

            line = 'Epoch [{}/{}], Pixel AUROC: {:.3f}, Sample AUROC: {:.3f}, Pixel AUPRO: {:.3f}, Best Threshold: {:.4f}, Best F1 Score: {:.4f}\n'.format(
                epoch + 1, epochs, auroc_px, auroc_sp, aupro_px, best_th, best_f1
            )
            print(line.strip())  # 打印
            log_file.write(line)  # <--- 写入txt文件

            torch.save({'decoder': decoder.state_dict()}, ckp_path)

    log_file.close()  # <--- 训练完后记得关闭txt文件！

    return auroc_px, auroc_sp, aupro_px, best_th, best_f1


if __name__ == '__main__':
    setup_seed(111)
    item_list = ['bread']
    # item_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']
    for i in item_list:
        train(i)
