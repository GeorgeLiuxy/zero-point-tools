import torch.nn as nn
from transformers import AutoModel, AutoConfig
DROPOUT = 0.3


class SimpleSpanishToxicClassifier(nn.Module):
    def __init__(self, num_labels=4, dropout=0.3):
        super().__init__()
        # 加载西语BERT模型
        self.ptm_path = "dccuchile/bert-base-spanish-wwm-uncased"
        config = AutoConfig.from_pretrained(self.ptm_path)

        config.update({
            "hidden_dropout_prob": dropout,
            "attention_probs_dropout_prob": dropout
        })
        self.bert = AutoModel.from_pretrained(self.ptm_path,config=config)

        # 简化分类头
        self.classifier = nn.Sequential(

            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.6),
            nn.Linear(512, num_labels)
        )


    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]标记作为分类特征
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取第一个标记([CLS])的输出
        logits = self.classifier(pooled_output)
        # 分类预测
        return {
            'logits': logits,
        }


