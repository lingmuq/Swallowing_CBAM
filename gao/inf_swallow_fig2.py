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
#
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2

from dataset1 import get_data_transforms, SwallowingDataset
import utils
import warnings
warnings.filterwarnings('ignore')

def get_mask(score_map, threshold):
    mask = copy.deepcopy(score_map)
    mask[score_map <= threshold] = 0
    mask[score_map > threshold] = 1
    return np.uint8(mask)

def get_contours(binary_img, thickness=2):
    binary_img[binary_img==1]=255
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, contours, -1, (255), thickness)
    return np.uint8(mask)

def mask_on_image(mask, image, alpha=0.8, color=[229, 149, 0]):
    img_ = copy.deepcopy(image)
    indices = np.where(mask == 255)
    mask_ = np.stack([mask] * 3, axis=2)/255
    mask_ = np.uint8(mask_*color)

    img_[indices] = img_[indices] * (1-alpha) + mask_[indices] * alpha
    return np.uint8(img_)

def mask_on_heatmap(mask, score, image, alpha=0.8):
    img_ = copy.deepcopy(image)
    indices = np.where(mask == 255)

    img_[indices] = img_[indices] * (1-alpha) + score[indices] * alpha
    return np.uint8(img_)
    

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

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
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


gpu = [0]
seed = 0
image_size = 256
batch_size = 1

class_list = ['soda_water']
# class_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']

#---Device setup------------------------------------#
set_device = utils.set_torch_device(gpu)
print("cuda:{0}".format(set_device))
#---------------------------------------------------#

#---Set random seed---------------------------------#
utils.fix_seeds(seed, set_device)
#---------------------------------------------------#

data_transform, gt_transform = get_data_transforms(image_size, image_size)

for _class in class_list:
    train_path = './1111/' + _class + '/train'
    test_path = './1111/' + _class

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SwallowingDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    encoder = encoder.to(set_device)
    bn = bn.to(set_device)
    decoder = decoder.to(set_device)

    ckp = torch.load('./checkpoints/wres50_bread.pth')
    for k, v in list(ckp['bn'].items()):
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
        for img, gt, label, _ in test_dataloader:
            img = img.to(set_device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list),3)

        gt_array = np.array(gt_list_px)
        pr_array = np.array(pr_list_px)
        precision, recall, thresholds = precision_recall_curve(gt_array.flatten(), pr_array.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
    print(_class,': AUROC-px=',auroc_px,',PRO=',aupro_px,',AUROC-im=',auroc_sp)
    print(threshold)

    # threshold = 0.393395272788271 # bread
    threshold = 0.393395272788271 # soda
    # threshold = 0.373395272788271 # for yogurt and soda
    count = 0
    alpha = 0.8

    thickness = 2
    alpha_heatmap=0.3
    alpha_contours=0.7
    color_pred=[0, 149, 229]
    color_gt=[0, 0, 255]

    if not os.path.exists('./results_all/'+_class):
        os.makedirs('./results_all/'+_class)
    if not os.path.exists('./results_all/'+_class+'/pred'):
        os.makedirs('./results_all/'+_class+'/pred')
    if not os.path.exists('./results_all/'+_class+'/gt'):
        os.makedirs('./results_all/'+_class+'/gt')
    if not os.path.exists('./results_all/'+_class+'/hmap'):
        os.makedirs('./results_all/'+_class+'/hmap')

    map_list = []
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            img = img.to(set_device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            map_list.append(anomaly_map)
    
    # for paper visualization of bread and empty category in Fig.2
    # norm_list = copy.deepcopy(map_list)
    # del norm_list[209:221]
    # val_max = np.max(norm_list)
    # val_min = np.min(norm_list)

    # for paper visualization of yogurt category in Fig.2
    # val_max = 0.6143
    # val_min = 0.0036

    # generally used it
    val_max = np.max(map_list)
    val_min = np.min(map_list)
    for img, gt, label, _ in test_dataloader:
        img = img.to(set_device)
        anomaly_map = map_list[count]

        # Prediction
        score = np.array(anomaly_map)
        score = (score - val_min) / (val_max - val_min + 1e-8)
        img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        img = np.uint8(min_max_norm(img)*255)
        mask_solid = get_mask(anomaly_map,threshold)
        mask_contours = get_contours(mask_solid,thickness)
        heatmap = cv2.applyColorMap(np.uint8(score*255), cv2.COLORMAP_JET)
        img_pred = mask_on_heatmap(mask_solid,heatmap,img,alpha_heatmap)
        img_pred = mask_on_image(mask_contours,img_pred,alpha_contours,color_pred)
        cv2.imwrite('./results_all/'+_class+'/pred/'+str(count)+'_'+'ad.png', img_pred)

        # Ground-truth
        gt = np.squeeze(gt.permute(0, 2, 3, 1).cpu().numpy()[0])
        gt = np.uint8(min_max_norm(gt)*255)
        mask_gt_contours = get_contours(gt,thickness)
        img_gt = mask_on_image(mask_gt_contours,img,alpha_contours,color_gt)
        cv2.imwrite('./results_all/'+_class+'/gt/'+str(count)+'_'+'ad.png', img_gt)

        # heatmap
        # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8) *255
        cv2.imwrite('./results_all/'+_class+'/hmap/'+str(count)+'_'+'ad.png', heatmap)


        # anomaly_map[anomaly_map <= threshold] = 0
        # ano_map = min_max_norm(anomaly_map)
        # ano_map = cvt2heatmap(ano_map*255)
        # img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        # img = np.uint8(min_max_norm(img)*255)

        # if not os.path.exists('./results_all/'+_class):
        #     os.makedirs('./results_all/'+_class)

        # ano_map = show_cam_on_image(img, ano_map)
        # cv2.imwrite('./results_all/'+_class+'/'+str(count)+'_'+'ad.png', ano_map)

        count += 1


