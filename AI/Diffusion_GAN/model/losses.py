import torch
import torch.nn.functional as F

# 对抗损失
def adversarial_loss(output, target_is_real):
    target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
    return F.binary_cross_entropy(output, target)

# L1损失
def l1_loss(output, target):
    return F.l1_loss(output, target)

# WGAN-GP 损失
def wgan_gp_loss(discriminator, real_image, fake_image, lambda_gp=10):
    batch_size = real_image.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(real_image.device)
    interpolated = epsilon * real_image + (1 - epsilon) * fake_image
    interpolated.requires_grad_()

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones_like(d_interpolated),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
