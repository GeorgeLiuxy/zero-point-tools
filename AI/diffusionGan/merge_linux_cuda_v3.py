import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# 超参数定义
num_epochs = 100
num_steps = 1000  # 扩散过程的时间步数
alpha = torch.linspace(0.0001, 0.02, num_steps)  # 正向扩散的方差
alpha_cumprod = torch.cumprod(1 - alpha, dim=0)  # 方差累积

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集定义
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
        ir_image = Image.open(os.path.join(self.ir_dir, self.ir_images[idx])).convert('RGB')  # 确保是 RGB 图像
        visible_image = Image.open(os.path.join(self.visible_dir, self.visible_images[idx])).convert('RGB')  # 确保是 RGB 图像

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

# 时间编码模块
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=6):  # 嵌入维度与图像通道数匹配
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(1, embed_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        t = t.unsqueeze(-1).float()  # [N] -> [N, 1]
        return self.linear2(self.act(self.linear1(t)))  # [N, embed_dim]


# Diffusion Generator
class DiffusionGenerator(nn.Module):
    def __init__(self):
        super(DiffusionGenerator, self).__init__()
        self.time_embed = TimeEmbedding(embed_dim=6)  # 时间嵌入的通道数与图像一致
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # 输入通道数为 6（红外 + 可见光图像拼接）
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

    def forward(self, x, t):
        # 时间嵌入
        t_embed = self.time_embed(t)  # [N, 6]
        t_embed = t_embed[:, :, None, None].repeat(1, 1, x.size(2), x.size(3))  # [N, 6, H, W]

        # 检查输入通道数
        if x.size(1) != 6:  # 确保输入是 6 通道
            raise ValueError(f"Expected input with 6 channels (IR + Visible), but got {x.size(1)} channels.")

        # 拼接时间嵌入
        x = x + t_embed  # 加法操作，而不是通道拼接
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# 判别器
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

# 正向扩散过程
def forward_diffusion(x, t, noise=None):
    if x.size(1) != 6:  # 检查通道数
        raise ValueError(f"Input to forward_diffusion must have 6 channels, but got {x.size(1)} channels.")
    if noise is None:
        noise = torch.randn_like(x)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t])[:, None, None, None]
    return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise



# 损失函数
def adversarial_loss(output, target_is_real):
    target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
    return nn.BCELoss()(output, target)

# 数据加载
train_dataset = IRVisibleDataset("ir_images", "visible_images", transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 初始化模型
generator = DiffusionGenerator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# 训练循环
for epoch in range(num_epochs):
    for i, (ir_image, visible_image) in enumerate(train_loader):
        # 检查加载后的图像形状
        # print(f"IR Image Shape: {ir_image.shape}")  # [N, 3, H, W]
        # print(f"Visible Image Shape: {visible_image.shape}")  # [N, 3, H, W]

        ir_image, visible_image = ir_image.to(device), visible_image.to(device)

        # 拼接红外图像和可见光图像
        combined_image = torch.cat((ir_image, visible_image), dim=1)  # [N, 6, H, W]
        # print(f"Combined Image Shape: {combined_image.shape}")  # 确认通道数为 6

        # 正向扩散
        t = torch.randint(0, num_steps, (combined_image.size(0),), device=device)
        noisy_image = forward_diffusion(combined_image, t)
        # 训练生成器
        optimizer_G.zero_grad()
        fake_image = generator(noisy_image, t)
        fake_output = discriminator(fake_image)
        g_loss = adversarial_loss(fake_output, True)
        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()
        real_output = discriminator(ir_image)
        fake_output = discriminator(fake_image.detach())
        d_loss = (adversarial_loss(real_output, True) + adversarial_loss(fake_output, False)) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # 保存生成图像
    if (epoch + 1) % 10 == 0:
        # 将生成图像值范围从 [-1, 1] 映射到 [0, 1]
        fake_image_normalized = (fake_image + 1) / 2
        save_image(fake_image_normalized, f"output/generated_image_epoch_{epoch+1}.png")


# 保存模型
torch.save(generator.state_dict(), "generator_model.pth")
torch.save(discriminator.state_dict(), "discriminator_model.pth")
