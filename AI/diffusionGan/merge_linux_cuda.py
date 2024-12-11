import torch.optim as optim
import os

import cv2  # 用于伪彩色和边缘增强
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19
from torchvision.transforms import ToPILImage


# 数据加载与预处理
class IRVisibleDataset(Dataset):
    def __init__(self, ir_dir, visible_dir, transform=None):
        self.ir_dir = ir_dir
        self.visible_dir = visible_dir
        self.transform = transform
        self.ir_images = sorted(os.listdir(ir_dir))
        self.visible_images = sorted(os.listdir(visible_dir))

    def __len__(self):
        return len(self.ir_images)

    def __getitem__(self, idx):
        ir_image = Image.open(os.path.join(self.ir_dir, self.ir_images[idx])).convert('RGB')
        visible_image = Image.open(os.path.join(self.visible_dir, self.visible_images[idx])).convert('RGB')

        if self.transform:
            ir_image = self.transform(ir_image)
            visible_image = self.transform(visible_image)

        return ir_image, visible_image

# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]范围
])

# 生成器（改进版，增加自注意力机制）
class DiffusionGenerator(nn.Module):
    def __init__(self):
        super(DiffusionGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Softmax(dim=1)  # 自注意力机制
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, ir_image, visible_image):
        x = torch.cat((ir_image, visible_image), dim=1)
        x = self.encoder(x)
        x = self.attention(x) * x
        x = self.decoder(x)
        return x

# 判别器（改进版，增加深度与批归一化）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg)[:12]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, fake_image, real_image):
        fake_features = self.feature_extractor(fake_image)
        real_features = self.feature_extractor(real_image)
        return nn.MSELoss()(fake_features, real_features)

# 损失函数
def adversarial_loss(output, target_is_real):
    target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
    return nn.BCELoss()(output, target)

def l1_loss(output, target):
    output_resized = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
    return nn.L1Loss()(output_resized, target)

# 保存对比图像
def save_comparison_image(ir_image, visible_image, fake_image, epoch):
    ir_image = transforms.ToPILImage()(ir_image.cpu().squeeze(0))
    visible_image = transforms.ToPILImage()(visible_image.cpu().squeeze(0))
    fake_image = transforms.ToPILImage()(fake_image.cpu().squeeze(0))

    def add_text_to_image(image, text):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10, 10), text, fill="white")
        return image

    ir_image = add_text_to_image(ir_image, "Infrared Image")
    visible_image = add_text_to_image(visible_image, "Visible Light Image")
    fake_image = add_text_to_image(fake_image, "Fused Image")

    combined_image = Image.new("RGB", (ir_image.width * 3, ir_image.height))
    combined_image.paste(ir_image, (0, 0))
    combined_image.paste(visible_image, (ir_image.width, 0))
    combined_image.paste(fake_image, (ir_image.width * 2, 0))
    combined_image.save(f"output/comparison_epoch_{epoch}.png")

# **伪彩色处理和细节增强函数**
def enhance_and_colorize(fused_image, visible_image, save_path):
    """
    对融合后的图像进行伪彩色处理和清晰化。
    :param fused_image: 生成的融合图像，已是 PIL.Image 格式
    :param visible_image: 可见光图像，已是 PIL.Image 格式
    :param save_path: 处理后的图像保存路径
    """
    # 1. 将融合图像和可见光图像从 PIL.Image 转换为 NumPy 格式
    fused_image_np = np.array(fused_image)  # 转换为 [H, W, C]
    visible_image_np = np.array(visible_image)  # 转换为 [H, W, C]

    # 2. 使用 OpenCV 提取可见光图像的细节（边缘检测）
    visible_gray = cv2.cvtColor(visible_image_np, cv2.COLOR_RGB2GRAY)  # 转为灰度图
    edges = cv2.Canny(visible_gray, threshold1=50, threshold2=150)  # 边缘检测
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)  # 映射为伪彩色

    # 3. 调整融合图像的对比度
    fused_image_pil = Image.fromarray(fused_image_np)
    enhancer = ImageEnhance.Contrast(fused_image_pil)
    fused_image_pil = enhancer.enhance(1.5)  # 增强对比度
    fused_image_np = np.array(fused_image_pil)

    # 4. 合并增强后的伪彩色与融合图像
    fused_image_colored = cv2.addWeighted(fused_image_np, 0.7, edges_colored, 0.3, 0)  # 加权合成

    # 5. 保存结果
    final_image = Image.fromarray(cv2.cvtColor(fused_image_colored, cv2.COLOR_BGR2RGB))  # 转回 RGB 格式
    final_image.save(save_path)
    print(f"Enhanced image saved at: {save_path}")



# **保存对比图像函数，加入伪彩色处理**
def save_comparison_image_with_enhancement(ir_image, visible_image, fake_image, epoch):
    """
    保存红外图像、可见光图像和融合图像的对比，并对融合图像进行伪彩色处理和增强。
    """
    ir_image = transforms.ToPILImage()(ir_image.cpu().squeeze(0))
    visible_image = transforms.ToPILImage()(visible_image.cpu().squeeze(0))
    fake_image = transforms.ToPILImage()(fake_image.cpu().squeeze(0))

    def add_text_to_image(image, text):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10, 10), text, fill="white")
        return image

    ir_image = add_text_to_image(ir_image, "Infrared Image")
    visible_image = add_text_to_image(visible_image, "Visible Light Image")
    fake_image = add_text_to_image(fake_image, "Fused Image")

    # 拼接图像
    combined_image = Image.new("RGB", (ir_image.width * 3, ir_image.height))
    combined_image.paste(ir_image, (0, 0))
    combined_image.paste(visible_image, (ir_image.width, 0))
    combined_image.paste(fake_image, (ir_image.width * 2, 0))
    comparison_path = f"output/comparison_epoch_{epoch}.png"
    combined_image.save(comparison_path)
    print(f"Comparison image saved at: {comparison_path}")

    # 调用伪彩色处理函数
    enhanced_image_path = f"output/enhanced_fused_image_epoch_{epoch}.png"
    enhance_and_colorize(fake_image, visible_image, enhanced_image_path)


if __name__ == '__main__':
    # 训练过程
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = DiffusionGenerator().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)

    train_dataset = IRVisibleDataset("ir_images", "visible_images", transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # **训练过程：调用伪彩色和细节增强**
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (ir_image, visible_image) in enumerate(train_loader):
            ir_image, visible_image = ir_image.to(device), visible_image.to(device)
            real_image = ir_image

            # 训练判别器
            fake_image = generator(ir_image, visible_image)
            optimizer_D.zero_grad()
            real_output = discriminator(real_image)
            fake_output = discriminator(fake_image.detach())
            d_loss = (adversarial_loss(real_output, True) + adversarial_loss(fake_output, False)) / 2
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_image)
            g_loss = adversarial_loss(fake_output, True) + 10 * l1_loss(fake_image, real_image) + 0.1 * perceptual_loss(fake_image, real_image)
            g_loss.backward()
            optimizer_G.step()

        scheduler_G.step()
        scheduler_D.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            ir_image_normalized = (ir_image + 1) / 2
            visible_image_normalized = (visible_image + 1) / 2
            fake_image_normalized = (fake_image + 1) / 2

            # 保存对比图像并增强融合图像
            save_comparison_image_with_enhancement(ir_image_normalized[0], visible_image_normalized[0], fake_image_normalized[0], epoch + 1)

    torch.save(generator.state_dict(), "generator_model.pth")
    torch.save(discriminator.state_dict(), "discriminator_model.pth")
