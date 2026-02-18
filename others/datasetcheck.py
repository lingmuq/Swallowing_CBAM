import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset_swallowing import SwallowingDataset, get_data_transforms


root = "./swallowing/bread"
size = 256
isize = 256
img_transform, gt_transform = get_data_transforms(size=size, isize=isize)
dataset = SwallowingDataset(
    root=root,
    transform=img_transform,
    gt_transform=gt_transform,
    phase='test'
)
bread_indices = [i for i, t in enumerate(dataset.types) if t == 'bread']
assert len(bread_indices) > 0, "breadない"
first_idx = bread_indices[0]
img, gt, label, img_type = dataset[first_idx]
mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
img_np = img.numpy()
img_np = img_np * std + mean
img_np = np.clip(img_np, 0, 1)
img_np = np.transpose(img_np, (1,2,0))
plt.figure(figsize=(5,5))
plt.imshow(img_np)
plt.title(f"bread first image (index={first_idx})")
plt.axis("off")
plt.show()
save_path = "bread_first_image.png"
plt.imsave(save_path, img_np)
print(f"Saved image to: {save_path}")
print("image shape :", img.shape)
print("GT shape    :", gt.shape)
print("label       :", label)
print("type        :", img_type)