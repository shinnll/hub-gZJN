import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ====================== 超参数 ======================
SEED = 42
N_SAMPLES_PER_CLASS = 1000   # 每类1000条，共5000条
MAXLEN = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 15

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ====================== 数据生成 ======================
def generate_samples(n_per_class=1000):
    chars = ['好', '的', '是', '我', '他', '她', '这', '那', '很', '真', '棒', '赞', '喜', '欢', '朋', '友', '今', '天', '啊', '吗']
    data = []
    for pos in range(5):  # 0~4 对应第1~5位
        for _ in range(n_per_class):
            sentence = [''] * 5
            sentence[pos] = '你'
            # 填充其他位置
            for i in range(5):
                if i != pos:
                    sentence[i] = random.choice(chars)
            text = ''.join(sentence)
            data.append((text, pos))  # (文本, 标签0-4)
    random.shuffle(data)
    return data

# ====================== 词表与编码 ======================
def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for s in sentences:
        for c in s:
            if c not in vocab:
                vocab[c] = idx
                idx += 1
    return vocab

def encode(text, vocab, maxlen=5):
    ids = [vocab.get(c, 1) for c in text]
    if len(ids) < maxlen:
        ids += [0] * (maxlen - len(ids))
    return ids[:maxlen]

# ====================== Dataset ======================
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        ids = encode(text, self.vocab, MAXLEN)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ====================== 模型定义 ======================
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)                    # (batch, seq_len, embed)
        rnn_out, _ = self.rnn(x)                 # (batch, seq_len, hidden)
        out = rnn_out[:, -1, :]                  # 取最后一个时间步
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

# ====================== 训练主流程 ======================
if __name__ == '__main__':
    # 生成数据
    print("正在生成训练数据...")
    data = generate_samples(N_SAMPLES_PER_CLASS)
    sentences = [item[0] for item in data]
    
    # 构建词表
    vocab = build_vocab(sentences)
    print(f"词表大小: {len(vocab)}")
    
    # 划分数据集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 模型、损失、优化器
    model = PositionRNN(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 训练
    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
    
    print("训练完成！最终验证准确率:", val_acc)
    
    # ====================== 测试 ======================
    print("\n=== 测试 ===")
    test_sentences = [
        "你今天很好",  # 第1位
        "好你今天很",  # 第2位
        "很好你今天",  # 第3位
        "今天你很好",  # 第4位
        "今天很好你",  # 第5位
        "我喜欢你啊",  # 第4位
        "大家你好吗",  # 第3位
    ]
    
    model.eval()
    for text in test_sentences:
        ids = torch.tensor([encode(text, vocab, MAXLEN)], dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
            pred_pos = logits.argmax(dim=1).item() + 1  # 转为1-5
        print(f"文本: {text} → 你在第 {pred_pos} 位")
