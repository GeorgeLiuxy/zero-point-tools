# train.py

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from AI.agent_prediction.code.utils import clip_gradients


def action_loss(predicted, target):
    """
    计算动作预测的损失。

    参数：
    - predicted：预测的动作，形状 (batch_size, pred_length, num_agents, 2)。
    - target：真实的动作，形状 (batch_size, pred_length, num_agents, 2)。

    返回：
    - 损失值。
    """
    loss = F.mse_loss(predicted, target)
    return loss


def compute_metrics(predicted, target):
    """
    计算评估指标：MAE、MSE、RMSE、MAPE。

    参数：
    - predicted：预测的动作，形状 (batch_size, pred_length, num_agents, 2)。
    - target：真实的动作，形状 (batch_size, pred_length, num_agents, 2)。

    返回：
    - metrics：包含 MAE、MSE、RMSE、MAPE 的字典。
    """
    with torch.no_grad():
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        mae = np.mean(np.abs(predicted - target))
        mse = np.mean((predicted - target) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((predicted - target) / (target + 1e-8))) * 100  # 避免除零

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'MAPE': 0}
    num_batches = len(dataloader)
    for batch in dataloader:
        node_features, adjacency_matrices, edge_features, target_actions, mask = batch
        # 将数据移动到设备
        for key in node_features:
            for t in range(len(node_features[key])):
                for agent_id in range(len(node_features[key][t])):
                    if isinstance(node_features[key][t][agent_id], np.ndarray):
                        node_features[key][t][agent_id] = torch.tensor(node_features[key][t][agent_id], dtype=torch.float32).to(device)
                    elif node_features[key][t][agent_id] is None:
                        node_features[key][t][agent_id] = None
        adjacency_matrices = adjacency_matrices.to(device)
        edge_features = edge_features.to(device)
        target_actions = target_actions.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        # 前向传播
        predicted_actions = model(node_features, adjacency_matrices, edge_features, mask)
        # 计算损失
        loss = action_loss(predicted_actions, target_actions)
        # 反向传播和优化
        loss.backward()
        clip_gradients(model, max_norm=5)
        optimizer.step()
        total_loss += loss.item()

        # 计算评估指标
        metrics = compute_metrics(predicted_actions, target_actions)
        for key in total_metrics:
            total_metrics[key] += metrics[key]

    scheduler.step()
    avg_loss = total_loss / num_batches
    avg_metrics = {key: total_metrics[key] / num_batches for key in total_metrics}
    return avg_loss, avg_metrics


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'MAPE': 0}
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            node_features, adjacency_matrices, edge_features, target_actions, mask = batch
            # 将数据移动到设备
            for key in node_features:
                for t in range(len(node_features[key])):
                    for agent_id in range(len(node_features[key][t])):
                        if isinstance(node_features[key][t][agent_id], np.ndarray):
                            node_features[key][t][agent_id] = torch.tensor(node_features[key][t][agent_id], dtype=torch.float32).to(device)
                        elif node_features[key][t][agent_id] is None:
                            node_features[key][t][agent_id] = None
            adjacency_matrices = adjacency_matrices.to(device)
            edge_features = edge_features.to(device)
            target_actions = target_actions.to(device)
            mask = mask.to(device)

            # 前向传播
            predicted_actions = model(node_features, adjacency_matrices, edge_features, mask)
            # 计算损失
            loss = action_loss(predicted_actions, target_actions)
            total_loss += loss.item()

            # 计算评估指标
            metrics = compute_metrics(predicted_actions, target_actions)
            for key in total_metrics:
                total_metrics[key] += metrics[key]

    avg_loss = total_loss / num_batches
    avg_metrics = {key: total_metrics[key] / num_batches for key in total_metrics}
    return avg_loss, avg_metrics


def plot_metrics(train_metrics, val_metrics, epochs, save_dir='plots'):
    """
    绘制评估指标的变化曲线。

    参数：
    - train_metrics：训练集上的指标列表，每个元素是一个字典。
    - val_metrics：验证集上的指标列表，每个元素是一个字典。
    - epochs：训练的总轮数。
    - save_dir：保存图像的目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics_keys = train_metrics[0].keys()
    for key in metrics_keys:
        plt.figure()
        train_values = [m[key] for m in train_metrics]
        val_values = [m[key] for m in val_metrics]
        plt.plot(range(1, epochs + 1), train_values, label=f'Train {key}')
        plt.plot(range(1, epochs + 1), val_values, label=f'Val {key}')
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(f'{key} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{key}_curve.png'))
        plt.close()
