import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
 #ひとつのカテゴリーすべでの画像の結果です
# 1. 设定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 加载模型
encoder, bn = wide_resnet50_2(pretrained=True)
decoder = de_wide_resnet50_2(pretrained=False)

encoder = encoder.to(device)
bn = bn.to(device)
decoder = decoder.to(device)

# 加载保存的权重
checkpoint = torch.load('./checkpoints/wres50_yokan.pth')
bn.load_state_dict(checkpoint['bn'])
decoder.load_state_dict(checkpoint['decoder'])

encoder.eval()
bn.eval()
decoder.eval()

# 3. 图片预处理
def preprocess_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # 增加batch维度
    return img.to(device)

def anomaly_map(input_tensor, original_size):
    with torch.no_grad():
        inputs = encoder(input_tensor)
        outputs = decoder(bn(inputs))

        cos_loss = torch.nn.CosineSimilarity(dim=1)
        score_map = 0
        for i in range(len(inputs)):
            target_size = inputs[0].shape[-1]
            input_i = F.interpolate(inputs[i], size=target_size, mode='bilinear', align_corners=True)
            output_i = F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=True)

            score_map += (1 - cos_loss(input_i, output_i))

        score_map = score_map / len(inputs)
        score_map = score_map.unsqueeze(1)  # (N=1, C=1, H, W)
        score_map = F.interpolate(score_map, size=original_size, mode='bilinear', align_corners=True)
        score_map = score_map.squeeze().cpu().numpy()
        score_map = (score_map - np.min(score_map)) / (np.max(score_map) - np.min(score_map) + 1e-8)
    return score_map

# 处理单张图片
def process_single_image(image_path, anomaly_save_dir, mask_save_dir, threshold, show=False):
    img = preprocess_image(image_path)
    img_show = Image.open(image_path).convert('RGB')
    original_size = img_show.size[::-1]  # (H, W)

    anomaly_score = anomaly_map(img, original_size)

    # 生成异常Mask
    mask = (anomaly_score > threshold).astype(np.uint8) * 255  # 白色为异常

    # 文件名处理
    filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename)[0]

    anomaly_save_path = os.path.join(anomaly_save_dir, filename_no_ext + '_anomaly.png')
    mask_save_path = os.path.join(mask_save_dir, filename_no_ext + '_mask.png')

    # 保存anomaly map和mask
    plt.imsave(anomaly_save_path, anomaly_score, cmap='jet')
    Image.fromarray(mask).save(mask_save_path)

    if show:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.axis('off')
        plt.imshow(img_show)

        plt.subplot(1, 3, 2)
        plt.title('Anomaly Map')
        plt.axis('off')
        plt.imshow(anomaly_score, cmap='jet')

        plt.subplot(1, 3, 3)
        plt.title('Anomaly Mask')
        plt.axis('off')
        plt.imshow(mask, cmap='gray')

        plt.show()

# 处理整个文件夹
def run_inference_folder(folder_path, threshold=0.7, show=False):
    # 保存目录
    anomaly_save_dir = os.path.join(folder_path, 'anomaly_score')
    mask_save_dir = os.path.join(folder_path, 'mask')

    # 自动创建保存目录
    os.makedirs(anomaly_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # 只处理常见图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.splitext(f)[-1].lower() in image_extensions]

    # 遍历每一张图
    for image_path in image_paths:
        process_single_image(image_path, anomaly_save_dir, mask_save_dir, threshold, show)

if __name__ == "__main__":
    # 输入你的文件夹路径
    folder_path = "C:\\Users\\18458\\Desktop\\meiyou" # 改成你的测试文件夹
    run_inference_folder(folder_path, threshold=0.7, show=False)  # threshold统一管理
