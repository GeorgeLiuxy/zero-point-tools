import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据加载与预处理

class IRVisibleDataset(Dataset):
    def __init__(self, ir_dir, visible_dir, transform=None):
        """
        初始化数据集
        :param ir_dir: 红外图像文件夹路径
        :param visible_dir: 可见光图像文件夹路径
        :param transform: 预处理操作
        """
        self.ir_dir = ir_dir
        self.visible_dir = visible_dir
        self.transform = transform
        self.ir_images = os.listdir(ir_dir)
        self.visible_images = os.listdir(visible_dir)

    def __len__(self):
        return len(self.ir_images)

    def __getitem__(self, idx):
        """
        获取一对红外和可见光图像
        """
        ir_image = Image.open(os.path.join(self.ir_dir, self.ir_images[idx]))
        visible_image = Image.open(os.path.join(self.visible_dir, self.visible_images[idx])).convert('RGB')

        # 如果红外图像是灰度图像，则将其转化为伪彩色图像
        if ir_image.mode != 'RGB':
            ir_image = ir_image.convert('RGB')  # 将灰度图转换为RGB格式

        if self.transform:
            ir_image = self.transform(ir_image)
            visible_image = self.transform(visible_image)

        return ir_image, visible_image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 2. 定义DiffusionGAN模型

class UNetDiffusion(nn.Module):
    def __init__(self):
        super(UNetDiffusion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1),
            nn.Tanh()  # 输出图像在[-1, 1]之间
        )

    def forward(self, ir_image, visible_image):
        x = torch.cat((ir_image, visible_image), dim=1)  # 拼接红外与可见光图像
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 3. 损失函数

def adversarial_loss(output, target_is_real):
    target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
    return nn.BCELoss()(output, target)

def l1_loss(output, target):
    return nn.L1Loss()(output, target)

# 4. 训练过程

# 定义设备：如果没有GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
generator = UNetDiffusion().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 加载数据集
train_dataset = IRVisibleDataset("ir_images", "visible_images", transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练
num_epochs = 50
for epoch in range(num_epochs):
    for i, (ir_image, visible_image) in enumerate(train_loader):
        ir_image = ir_image.to(device)
        visible_image = visible_image.to(device)

        # 训练判别器
        real_image = ir_image  # 假设红外图像为真实图像
        fake_image = generator(ir_image, visible_image)

        # 判别器对真实图像的判别
        optimizer_D.zero_grad()
        real_output = discriminator(real_image)
        d_loss_real = adversarial_loss(real_output, True)

        # 判别器对生成图像的判别
        fake_output = discriminator(fake_image.detach())  # 不更新生成器
        d_loss_fake = adversarial_loss(fake_output, False)

        # 判别器总损失
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_image)
        g_loss = adversarial_loss(fake_output, True)  # 生成器希望判别器认为生成图像是真的
        reconstruction_loss = l1_loss(fake_image, real_image)  # L1损失
        g_loss = g_loss + 100 * reconstruction_loss  # 综合损失

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# 5. 结果可视化

def show_images(real_image, fake_image):
    # 反归一化处理
    real_image = (real_image + 1) / 2  # 由于使用Tanh，所以需要反归一化
    fake_image = (fake_image + 1) / 2

    real_image = real_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_image = fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(real_image[0])
    axes[0].set_title("Real Image")
    axes[1].imshow(fake_image[0])
    axes[1].set_title("Fake Image")
    plt.show()

# def show_images(ir_image, visible_image, fake_image):
#     """
#     展示红外图像、可视光图像和融合后的图像
#     """
#     # 反归一化处理
#     ir_image = (ir_image + 1) / 2  # 由于使用Tanh，所以需要反归一化
#     visible_image = (visible_image + 1) / 2
#     fake_image = (fake_image + 1) / 2
#
#     ir_image = ir_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
#     visible_image = visible_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
#     fake_image = fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 修改为3列
#     axes[0].imshow(ir_image[0])  # 展示红外图像
#     axes[0].set_title("Infrared Image")
#     axes[1].imshow(visible_image[0])  # 展示可见光图像
#     axes[1].set_title("Visible Image")
#     axes[2].imshow(fake_image[0])  # 展示融合后的图像
#     axes[2].set_title("Fused Image")
#
#     plt.show()


# 示例：展示一张红外图像、可见光图像和生成图像的对比
show_images(ir_image, visible_image, fake_image)
