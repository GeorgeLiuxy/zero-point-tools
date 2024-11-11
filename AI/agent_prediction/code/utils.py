import torch

def adjust_learning_rate(optimizer, epoch, total_epochs, initial_lr):
    """
    使用余弦退火调整学习率。
    """
    lr = 0.5 * initial_lr * (1 + torch.cos(torch.tensor(epoch / total_epochs * 3.1415926535)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr.item()

def clip_gradients(model, max_norm):
    """
    梯度裁剪。
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

