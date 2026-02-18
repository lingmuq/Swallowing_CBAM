import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
#
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2

# 1. 设定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 加载模型
encoder, bn = wide_resnet50_2(pretrained=True)
decoder = de_wide_resnet50_2(pretrained=False)

# 放到GPU或CPU
encoder = encoder.to(device)
bn = bn.to(device)
decoder = decoder.to(device)

# 加载保存的权重
checkpoint = torch.load('./checkpoints/wres50_bread.pth')
bn.load_state_dict(checkpoint['bn'])
decoder.load_state_dict(checkpoint['decoder'])

# 设置为eval模式
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



import os

def run_inference(image_path, threshold=0.313395272788271):
    img = preprocess_image(image_path)
    img_show = Image.open(image_path).convert('RGB')
    original_size = img_show.size[::-1]  # (H, W)

    anomaly_score = anomaly_map(img, original_size)

    # 生成异常Mask
    mask = (anomaly_score > threshold).astype(np.uint8) * 255  # 白色为异常，黑色为正常

    # 🔥 正确保存路径，不修改原图
    filename = os.path.basename(image_path)        # 提取原图文件名，比如 "xxx.png"
    filename_no_ext = os.path.splitext(filename)[0] # 去掉扩展名，比如 "xxx"
    save_mask_path = os.path.join('./', filename_no_ext + '_mask.png')

    # 确保保存目录存在
    os.makedirs('./', exist_ok=True)

    Image.fromarray(mask).save(save_mask_path)

    print(f"Mask saved at: {save_mask_path}")

    # 同时显示原图+异常热力图+mask
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



# 6. 运行推理
if __name__ == "__main__":
    # 传入你想推理的图片路径
    image_path = './1111/bread/test/bread/009.bmp'  # 改成你的图片路径！
    run_inference(image_path)
