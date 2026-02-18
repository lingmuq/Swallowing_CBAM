import os
import copy
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from PIL import Image
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from statistics import mean
from skimage import measure
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset_swallowing import get_data_transforms, SwallowingDataset
import utils
import warnings
warnings.filterwarnings('ignore')


def get_mask(score_map, threshold):
    mask = copy.deepcopy(score_map)
    mask[score_map <= threshold] = 0
    mask[score_map > threshold] = 1
    return np.uint8(mask)


def get_contours(binary_img, thickness=2):
    binary_img[binary_img == 1] = 255
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, contours, -1, (255), thickness)
    return np.uint8(mask)


def mask_on_image(mask, image, alpha=0.8, color=[229, 149, 0]):
    img_ = copy.deepcopy(image)
    indices = np.where(mask == 255)
    mask_ = np.stack([mask] * 3, axis=2) / 255
    mask_ = np.uint8(mask_ * color)
    img_[indices] = img_[indices] * (1-alpha) + mask_[indices] * alpha
    return np.uint8(img_)


def mask_on_heatmap(mask, score, image, alpha=0.8):
    img_ = copy.deepcopy(image)
    indices = np.where(mask == 255)
    img_[indices] = img_[indices] * (1-alpha) + score[indices] * alpha
    return np.uint8(img_)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_max + 1e-8)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200):
    assert isinstance(amaps, ndarray)
    assert isinstance(masks, ndarray)
    assert amaps.ndim == masks.ndim == 3
    assert amaps.shape == masks.shape
    assert set(masks.flatten()) == {0, 1}
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                coords0 = region.coords[:, 0]
                coords1 = region.coords[:, 1]
                tp = binary_amap[coords0, coords1].sum()
                pros.append(tp / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th},
                       ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    return auc(df["fpr"], df["pro"])


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a = 1 - F.cosine_similarity(fs, ft)
        a = torch.unsqueeze(a, dim=1)
        a = F.interpolate(a, size=out_size,
                          mode='bilinear', align_corners=True)
        a = a[0, 0, :, :].cpu().numpy()
        a_map_list.append(a)
        if amap_mode == 'mul':
            anomaly_map *= a
        else:
            anomaly_map += a
    return anomaly_map, a_map_list


gpu = [0]
seed = 0
image_size = 256
batch_size = 1
class_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']
device = utils.set_torch_device(gpu)
utils.fix_seeds(seed, device)
data_transform, gt_transform = get_data_transforms(image_size, image_size)
for _class in class_list:
    print("=== Processing:", _class, "===")
    train_path = './swallowing/' + _class + '/train'
    test_path = './swallowing/' + _class
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SwallowingDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    weight_path = f'./checkpoints/wres50_{_class}.pth'
    ckp = torch.load(weight_path)
    for k, _ in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    encoder.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _ in test_loader:
            img = img.to(device)
            fs = encoder(img)
            ft = decoder(bn(fs))
            amap, _ = cal_anomaly_map(fs, ft, img.shape[-1], amap_mode='a')
            amap = gaussian_filter(amap, sigma=4)
            gt_bin = gt.clone()
            gt_bin[gt_bin > 0.5] = 1
            gt_bin[gt_bin <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(compute_pro(gt_bin.squeeze(0).cpu().numpy().astype(int), amap[np.newaxis, :, :]))
            gt_list_px.extend(gt_bin.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(amap.ravel())
            gt_list_sp.append(int(np.max(gt_bin.cpu().numpy())))
            pr_list_sp.append(float(np.max(amap)))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list), 3)
        print(_class, ': Pixel Auroc =', auroc_px, ', Sample Auroc =', auroc_sp, ', Pixel Aupro =', aupro_px)
        gt_arr = np.array(gt_list_px)
        pr_arr = np.array(pr_list_px)
        precision, recall, thresholds = precision_recall_curve(gt_arr.flatten(), pr_arr.flatten())
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        threshold = thresholds[np.argmax(f1)]
        print("Threshold =", threshold)
    out_dir = f'./results_all/{_class}'
    os.makedirs(out_dir + '/pred', exist_ok=True)
    os.makedirs(out_dir + '/gt', exist_ok=True)
    os.makedirs(out_dir + '/hmap', exist_ok=True)
    map_list = []
    with torch.no_grad():
        for img, gt, label, _ in test_loader:
            img = img.to(device)
            fs = encoder(img)
            ft = decoder(bn(fs))
            amap, _ = cal_anomaly_map(fs, ft, img.shape[-1], amap_mode='a')
            amap = gaussian_filter(amap, sigma=4)
            map_list.append(amap)
    val_max = np.max(map_list)
    val_min = np.min(map_list)
    count = 0
    thickness = 2
    alpha_heatmap = 0.3
    alpha_contours = 0.7
    color_pred = [0, 149, 229]
    color_gt = [0, 0, 255]
    for img, gt, label, _ in test_loader:
        img_np = img.to(device)
        fs = encoder(img_np)
        amap = map_list[count]
        score = (amap - val_min) / (val_max - val_min + 1e-8)
        img_vis = cv2.cvtColor(img_np.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        img_vis = np.uint8(min_max_norm(img_vis) * 255)
        mask_solid = get_mask(amap, threshold)
        mask_contours = get_contours(mask_solid, thickness)
        heatmap = cv2.applyColorMap(np.uint8(score * 255), cv2.COLORMAP_JET)
        img_pred = mask_on_heatmap(mask_solid, heatmap, img_vis, alpha_heatmap)
        img_pred = mask_on_image(mask_contours, img_pred, alpha_contours, color_pred)
        cv2.imwrite(f'./results_all/{_class}/pred/{count}_ad.png', img_pred)
        gt_np = np.squeeze(gt.permute(0, 2, 3, 1).cpu().numpy()[0])
        gt_np = np.uint8(min_max_norm(gt_np) * 255)
        gt_cont = get_contours(gt_np, thickness)
        img_gt = mask_on_image(gt_cont, img_vis, alpha_contours, color_gt)
        cv2.imwrite(f'./results_all/{_class}/gt/{count}_ad.png', img_gt)
        cv2.imwrite(f'./results_all/{_class}/hmap/{count}_ad.png', heatmap)
        count += 1
print("=== All classes finished ===")