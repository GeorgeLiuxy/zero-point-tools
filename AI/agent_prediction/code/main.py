# main.py

import argparse
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data_processing import AgentDataset
from models.stgnn import STGNN
from train import train_epoch, evaluate, plot_metrics
from AI.agent_prediction.code.utils import adjust_learning_rate
import os
import logging


def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ST-GNN Training')
    parser.add_argument('--data_dir', type=str, default='simulation_data', help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=8, help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='Prediction length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--history_length', type=int, default=10, help='Length of the history sequence')
    parser.add_argument('--map_size', type=int, default=11, help='Size of the local map')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models and plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = AgentDataset(data_dir=args.data_dir, seq_length=args.seq_length, pred_length=args.pred_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    # 节点特征维度：位置(2)、相对目标位置(2)、历史轨迹编码(hidden_dim)、局部地图编码(hidden_dim)
    node_feature_dim = 2 + 2 + args.hidden_dim + args.hidden_dim
    edge_feature_dim = 4  # 距离，角度，信噪比，p_los
    num_agents = None  # 可根据数据集动态获取
    for batch in dataloader:
        node_features, _, _, _, _ = batch
        num_agents = len(node_features[0]['positions'])
        break

    model = STGNN(node_feature_dim=node_feature_dim,
                  edge_feature_dim=edge_feature_dim,
                  hidden_dim=args.hidden_dim,
                  out_dim=2,  # 动作维度（速度、转向角）
                  num_heads=args.num_heads,
                  num_layers=args.num_layers,
                  seq_length=args.seq_length,
                  pred_length=args.pred_length,
                  device=device).to(device)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 记录指标
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    # 训练循环
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        train_loss, train_metric = train_epoch(model, dataloader, optimizer, scheduler, device)
        val_loss, val_metric = evaluate(model, dataloader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logging.info(f'Train Metrics: {train_metric}')
        logging.info(f'Val Metrics: {val_metric}')

        # 保存模型
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(args.save_dir, f'stgnn_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f'Model saved to {model_path}')

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    plt.close()

    # 绘制评估指标曲线
    plot_metrics(train_metrics, val_metrics, args.epochs, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
