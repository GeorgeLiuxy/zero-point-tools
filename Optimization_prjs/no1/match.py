import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from load_data import create_dataloaders
from model1 import SimpleSpanishToxicClassifier

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

class ToxicCommentTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, device,
                 model_save_dir='checkpoints/', num_epochs=10, early_stop_patience=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.criterion = nn.CrossEntropyLoss()
        os.makedirs(model_save_dir, exist_ok=True)


    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            # 数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            # 添加.float()转换
            labels = batch['labels'].to(self.device)
            labels = labels.to(torch.long)
            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            ##对比学习的特征

            # 计算损失
            cls_loss = self.criterion(logits, labels)

            # 反向传播
            cls_loss.backward()
            self.optimizer.step()

            # 统计信息
            total_loss += cls_loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)



            # 更新进度条
            progress_bar.set_postfix({
                'cls_loss': f"{cls_loss.item():.4f}",
                'acc': f"{correct / total_samples:.4f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def train(self):
        best_valid_acc = 0.0
        early_stop_counter = 0
        writer = SummaryWriter(self.model_save_dir)

        for epoch in range(self.num_epochs):
            # 训练阶段

            train_loss, train_acc = self.train_epoch(epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)

            # 验证阶段
            valid_loss, valid_acc = self.evaluate(self.valid_loader)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('Accuracy/valid', valid_acc, epoch)

  

            # 保存最佳模型
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_save_dir, 'best_model.pt'))
                early_stop_counter = 0
                print(f"New best model saved with accuracy {valid_acc:.4f}")
            else:
                early_stop_counter += 1

            # 早停检查
            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # 打印日志
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} | Acc: {valid_acc:.4f}\n")

        writer.close()
        print("Training completed.")


# 使用示例
if __name__ == "__main__":
    # 初始化组件
    device = "cuda"
    model = SimpleSpanishToxicClassifier(num_labels=45).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # 假设已经定义好数据加载器
    train_loader, valid_loader = create_dataloaders()
    # 初始化训练器
    trainer = ToxicCommentTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        model_save_dir='out/',
        num_epochs=10,
        early_stop_patience=3
    )
    # 开始训练
    trainer.train()



