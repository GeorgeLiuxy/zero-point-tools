import json

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 配置参数
MAX_LENGTH = 128
BATCH_SIZE = 32
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
TEST_SIZE = 0.2  # 验证集比例

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ret_maps = {}
with open("data/lable2i.json") as f:
    a2i = json.load(f)
    ret_maps["lable2i"] = a2i
    ret_maps["i2lable"] = {v: k for k, v in a2i.items()}


##标签传id
def label2idx(label, map):
    for i in range(len(label)):
        label[i] = map[str(label[i])]  # 直接操作一维元素
    return label




# 自定义数据集类
class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item




# 数据加载函数
def load_data():
    # 加载训练数据
    data_path="data/train.csv"

    train_df = pd.read_csv(data_path)
    texts = train_df['text'].values

    #加载标签
    labels = train_df['label'].values  # 假设列名为'label'
    labels = label2idx(labels, ret_maps["lable2i"])
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        stratify=labels,  # 保持类别分布
        random_state=42
    )


    return (X_train, y_train), (X_val, y_val)


# 创建数据加载器
def create_dataloaders():
    # 加载数据
    (X_train, y_train), (X_val, y_val) = load_data()
    # 创建数据集
    train_dataset = ToxicCommentDataset(X_train, y_train)
    val_dataset = ToxicCommentDataset(X_val, y_val)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader

def print_class_distribution(dataset, title, label_map):
    """打印类别分布（兼容NumPy数组）"""
    labels = dataset.labels  # 直接获取NumPy数组
    class_counts = labels.sum(axis=0)  # 按列求和
    print(f"\n{title}类别分布：")
    for idx, count in enumerate(class_counts):
        label_name = label_map[idx]
        print(f"类别 {label_name}（ID {idx}）: {int(count)} 样本")
# ----------------------------------------------------
# 使用示例
# ----------------------------------------------------
if __name__ == "__main__":


    # 初始化数据加载器
    train_loader, val_loader = create_dataloaders()

    # 查看一个训练批次
    sample_batch = next(iter(train_loader))
    print("\n训练批次结构：")
    print(f"Input IDs形状：{sample_batch['input_ids'].shape}")
    print(f"Attention Mask形状：{sample_batch['attention_mask'].shape}")
    print(f"Labels形状：{sample_batch['labels'].shape}")

    # 查看数据统计信息
    print("\n数据统计：")
    print(f"训练样本数：{len(train_loader.dataset)}")
    print(f"验证样本数：{len(val_loader.dataset)}")


#
# # 打印训练集和验证集的类别分布
#     print_class_distribution(train_loader.dataset, "训练集", ret_maps["i2lable"])
#     print_class_distribution(val_loader.dataset, "验证集", ret_maps["i2lable"])
