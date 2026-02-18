import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter
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
    mask = score_map.copy()
    mask[score_map <= threshold] = 0
    mask[score_map > threshold] = 1
    return np.uint8(mask)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
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
        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map


gpu = [0]
seed = 0
image_size = 256
batch_size = 1
class_list = ['bread', 'cracker', 'jelly', 'pudding', 'soda_water', 'yogurt', 'yokan']
set_device = utils.set_torch_device(gpu)
print("cuda:{0}".format(set_device))
utils.fix_seeds(seed, set_device)
data_transform, gt_transform = get_data_transforms(image_size, image_size)
df_results = pd.DataFrame([], columns=["class","Pixel Auroc","Sample Auroc","Pixel Aupro"])
for _class in class_list:
    print("Processing:", _class)
    train_path = './swallowing/' + _class + '/train'
    test_path = './swallowing/' + _class
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = SwallowingDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, _ = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder = encoder.to(set_device)
    decoder = decoder.to(set_device)
    ckp_path = f'./checkpoints/wres50_{_class}.pth'
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'])
    encoder.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            img = img.to(set_device)
            inputs, feature_c = encoder(img)
            outputs = decoder(feature_c)
            anomaly_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int), anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    aupro_px = round(np.mean(aupro_list), 3)
    df_results = df_results.append({
        "class": _class,
        "Pixel Auroc": auroc_px,
        "Sample Auroc": auroc_sp,
        "Pixel Aupro": aupro_px
    }, ignore_index=True)
df_results.to_csv('./eva_swallowing.csv', index=False)
print("Evaluation finished. Results saved to eva_swallowing.csv")
print(df_results)