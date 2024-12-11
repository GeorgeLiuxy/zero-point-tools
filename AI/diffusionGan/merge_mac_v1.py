import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# 数据加载与预处理
class IRVisibleDataset(Dataset):
    def __init__(self, ir_dir, visible_dir, transform=None):
        self.ir_dir = ir_dir
        self.visible_dir = visible_dir
        self.transform = transform
        self.ir_images = os.listdir(ir_dir)
        self.visible_images = os.listdir(visible_dir)

    def __len__(self):
        return len(self.ir_images)

    def __getitem__(self, idx):
        ir_image = Image.open(os.path.join(self.ir_dir, self.ir_images[idx])).convert('RGB')
        visible_image = Image.open(os.path.join(self.visible_dir, self.visible_images[idx])).convert('RGB')

        if self.transform:
            ir_image = self.transform(ir_image)
            visible_image = self.transform(visible_image)

        return ir_image, visible_image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 扩散模型生成器的定义
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
        self.diffusion_layer = nn.Conv2d(256, 256, 3, padding=1)  # 扩散层，用于模拟扩散过程
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
        x = self.diffusion_layer(x)  # 通过扩散层
        x = self.decoder(x)
        return x

# 判别器网络（使用传统GAN判别器结构）
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

# 损失函数：生成对抗损失（对抗损失 + L1损失）
def adversarial_loss(output, target_is_real):
    target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
    return nn.BCELoss()(output, target)

def l1_loss(output, target):
    output_resized = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
    return nn.L1Loss()(output_resized, target)

# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = DiffusionGenerator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

train_dataset = IRVisibleDataset("data/ir_images", "data/visible_images", transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    for i, (ir_image, visible_image) in enumerate(train_loader):
        ir_image = ir_image.to(device)
        visible_image = visible_image.to(device)

        # 训练判别器
        real_image = ir_image
        fake_image = generator(ir_image, visible_image)

        optimizer_D.zero_grad()
        real_output = discriminator(real_image)
        d_loss_real = adversarial_loss(real_output, True)

        fake_output = discriminator(fake_image.detach())
        d_loss_fake = adversarial_loss(fake_output, False)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_image)
        g_loss = adversarial_loss(fake_output, True)
        reconstruction_loss = l1_loss(fake_image, real_image)  # L1损失

        g_loss = g_loss + 100 * reconstruction_loss  # 综合损失
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# 结果可视化
def show_images(real_image, fake_image):
    real_image = (real_image + 1) / 2
    fake_image = (fake_image + 1) / 2

    real_image = real_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_image = fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(real_image[0])
    axes[0].set_title("Real Image")
    axes[1].imshow(fake_image[0])
    axes[1].set_title("Fake Image")
    plt.show()

# 展示融合效果
show_images(ir_image, fake_image)
