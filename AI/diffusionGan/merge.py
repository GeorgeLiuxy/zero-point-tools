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
        ir_image = Image.open(os.path.join(self.ir_dir, self.ir_images[idx])).convert('RGB')
        visible_image = Image.open(os.path.join(self.visible_dir, self.visible_images[idx])).convert('RGB')

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

# 加载数据集
# 修改为实际的路径
train_dataset = IRVisibleDataset("data/ir_images", "data/visible_images", transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


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
            nn.Tanh()
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

# 初始化模型和优化器
generator = UNetDiffusion().cuda()
discriminator = Discriminator().cuda()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练
num_epochs = 50
for epoch in range(num_epochs):
    for i, (ir_image, visible_image) in enumerate(train_loader):
        ir_image = ir_image.cuda()
        visible_image = visible_image.cuda()

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
    real_image = real_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_image = fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(real_image[0])
    axes[0].set_title("Real Image")
    axes[1].imshow(fake_image[0])
    axes[1].set_title("Fake Image")
    plt.show()

# 示例：展示一张真实图像和生成图像的对比
# 注意：这里使用生成器输出的图像和真实图像进行对比
show_images(ir_image, fake_image)
