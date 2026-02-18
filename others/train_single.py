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
from test import evaluation, visualization, test
from torch.nn import functional as F
import json


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


def train(target_class):
    print(target_class)
    epochs = 10
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results_epoch', exist_ok=True)
    log_path = f'./results_epoch/{target_class}.txt'
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = f'./swallowing/{target_class}/train'
    test_path = f'./swallowing/{target_class}'
    ckp_path = f'./checkpoints/wres50_{target_class}.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SwallowingDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    epoch_logs = {
        "epochs": [],
        "pixel_auroc": [],
        "sample_auroc": [],
        "pixel_aupro": []
    }
    with open(log_path, 'w') as f:
        for epoch in range(epochs):
            bn.train()
            decoder.train()
            loss_list = []
            for img, label in train_dataloader:
                img = img.to(device)
                inputs = encoder(img)
                outputs = decoder(bn(inputs))
                loss = loss_fucntion(inputs, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            log_line = f"epoch [{epoch + 1}/{epochs}], loss:{np.mean(loss_list):.4f}\n"
            print(log_line.strip())
            f.write(log_line)
            if (epoch + 1) % 10 == 0:
                auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
                result_line = (
                    f"Pixel Auroc:{auroc_px:.3f}, "
                    f"Sample Auroc:{auroc_sp:.3f}, "
                    f"Pixel Aupro:{aupro_px:.3f}\n"
                )
                print(result_line.strip())
                f.write(result_line)
                epoch_logs["epochs"].append(epoch + 1)
                epoch_logs["pixel_auroc"].append(auroc_px)
                epoch_logs["sample_auroc"].append(auroc_sp)
                epoch_logs["pixel_aupro"].append(aupro_px)
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
    os.makedirs("results_eva_json", exist_ok=True)
    out_json = f"results_eva_json/{target_class}.json"
    with open(out_json, "w") as mf:
        json.dump(epoch_logs, mf)
    return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':
    setup_seed(111)
    target_class = 'bread'
    start = time.time()
    train(target_class)
    end = time.time()
    print(f"----- time -----\n{target_class}: {(end - start)/60:.2f} minutes\n")