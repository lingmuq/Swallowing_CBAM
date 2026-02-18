import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import time
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset_swallowing import SwallowingDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
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
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
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
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results_epoch', exist_ok=True)
    log_path = f'./results_epoch/{_class_}.txt'
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    # train_path = './mvtec/' + _class_ + '/train'
    # test_path = './mvtec/' + _class_
    train_path = './swallowing/' + _class_ + '/train'
    test_path = './swallowing/' + _class_
    ckp_path = './checkpoints/' + 'wres50_' + _class_ + '.pth'
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
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    epochs_list = []
    pixel_auroc_list = []
    sample_auroc_list = []
    pixel_aupro_list = []
    with open(log_path, 'w') as f:
        for epoch in range(epochs):
            bn.train()
            decoder.train()
            loss_list = []
            for img, label in train_dataloader:
                img = img.to(device)
                inputs, s_cbam = encoder(img)
                outputs, s2_cbam = decoder(bn(inputs)) 
                loss = loss_fucntion(inputs, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            # print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
            log_line = 'epoch [{}/{}], loss:{:.4f}\n'.format(epoch + 1, epochs, np.mean(loss_list))
            print(log_line.strip())
            f.write(log_line)
            if (epoch + 1) % 10 == 0:
                auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
                # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
                result_line = 'Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3f}\n'.format(auroc_px, auroc_sp, aupro_px)
                print(result_line.strip())
                f.write(result_line)
                epochs_list.append(epoch + 1)
                pixel_auroc_list.append(auroc_px)
                sample_auroc_list.append(auroc_sp)
                pixel_aupro_list.append(aupro_px)
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
        import json
        eva_path = os.path.join("results_eva_json", f"{_class_}.json")
        with open(eva_path, "w") as mf:
            json.dump({
                "epochs": epochs_list,
                "pixel_auroc": pixel_auroc_list,
                "sample_auroc": sample_auroc_list,
                "pixel_aupro": pixel_aupro_list
            }, mf)
    return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':
    setup_seed(111)
    item_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']
    results = {}
    total_start = time.time()
    for i in item_list:
        class_start = time.time()
        train(i)
        class_end = time.time()
        elapsed_class = class_end - class_start
        results[i] = elapsed_class
    total_end = time.time()
    elapsed_total = total_end - total_start
    # print("----- time -----")
    # for cls, t in results.items():
    #     print(f"{cls:12s}: {t/60:.2f} minutes")
    # print(f"total: {elapsed_total/3600:.2f} hours")
    lines = []
    lines.append("----- time -----\n")
    for cls, t in results.items():
        lines.append(f"{cls:12s}: {t/60:.2f} minutes\n")
    lines.append(f"total: {elapsed_total/3600:.2f} hours\n")
    print("".join(lines))
    with open("time_results.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)