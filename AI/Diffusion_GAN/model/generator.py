import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionGenerator(nn.Module):
    def __init__(self):
        super(DiffusionGenerator, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        # 扩散层
        self.diffusion_layer = nn.Conv2d(256, 256, 3, padding=1)
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, ir_image, visible_image):
        # 拼接红外图像和可见光图像
        x = torch.cat((ir_image, visible_image), dim=1)  # 拼接两个输入图像，维度为(batch_size, 6, height, width)
        x = self.encoder(x)
        x = self.diffusion_layer(x)  # 扩散过程模拟
        x = self.decoder(x)
        return x
