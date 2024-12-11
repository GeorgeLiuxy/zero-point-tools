# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.generator import DiffusionGenerator
from model.discriminator import Discriminator
from model.losses import adversarial_loss, l1_loss, wgan_gp_loss
from dataset.dataset import IRVisibleDataset  # 确保路径正确


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
generator = DiffusionGenerator().to(device)
discriminator = Discriminator().to(device)

# 初始化优化器
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 数据加载
train_dataset = IRVisibleDataset("data/ir_images", "data/visible_images", transform=None)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练过程
num_epochs = 100
lambda_gp = 10  # 梯度惩罚的系数
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

        gp_loss = wgan_gp_loss(discriminator, real_image, fake_image, lambda_gp)
        d_loss = (d_loss_real + d_loss_fake) / 2 + lambda_gp * gp_loss
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
