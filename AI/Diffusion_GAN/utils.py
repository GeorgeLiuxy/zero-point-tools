import matplotlib.pyplot as plt
import numpy as np
import torch

# 展示图像
def show_images(real_image, fake_image):
    real_image = (real_image + 1) / 2  # 将[-1, 1]范围的图像转换为[0, 1]
    fake_image = (fake_image + 1) / 2

    real_image = real_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_image = fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(real_image[0])
    axes[0].set_title("Real Image")
    axes[1].imshow(fake_image[0])
    axes[1].set_title("Fake Image")
    plt.show()

# 其他辅助工具可以根据需要添加，如日志记录、模型保存等。
