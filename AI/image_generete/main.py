import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),  # 输入噪声的维度是100
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),  # 输出28x28像素的图像
            nn.Tanh()  # 输出范围[-1, 1]之间的像素值
        )

    def forward(self, z):
        return self.fc(z).view(-1, 1, 28, 28)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # 输出一个概率值
            nn.Sigmoid()  # 输出0到1之间的概率值
        )

    def forward(self, x):
        return self.fc(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将图像标准化到[-1, 1]范围
])

train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# 创建生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 优化器
lr = 0.0002
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 损失函数
criterion = nn.BCELoss()


num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # 训练判别器
        real_images = images
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)  # 真实图像标签为1
        fake_labels = torch.zeros(batch_size, 1)  # 假图像标签为0

        # 判别器输出
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, 100)  # 随机噪声
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())  # 生成器输出，判别器不需要反向传播
        loss_fake = criterion(output_fake, fake_labels)

        loss_d = loss_real + loss_fake

        # 优化判别器
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器
        output_fake = discriminator(fake_images)
        loss_g = criterion(output_fake, real_labels)  # 生成器希望判别器认为假图像为真

        # 优化生成器
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    # 每个epoch生成一些图片
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            noise = torch.randn(16, 100)
            generated_images = generator(noise)
            generated_images = generated_images.view(16, 28, 28).cpu().numpy()
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for ax, img in zip(axes.flatten(), generated_images):
                ax.imshow(img, cmap='gray')
                ax.axis('off')
            plt.show()
