import torch
import torch.nn as nn
import math
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embed(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)
        logits = self.fc(x)
        return logits

def greedy_generate(model, tokenizer, device, prompt, max_length=100):
    """
    贪心采样生成文本
    """
    model.eval()
    char2idx, idx2char = tokenizer['char2idx'], tokenizer['idx2char']
    
    # 1. 将提示词转换为模型能看懂的索引
    prompt_ids = [char2idx.get(c, char2idx.get('<unk>', 0)) for c in prompt if c in char2idx]
    if not prompt_ids:
        print("提示词包含太多未知字符，请更换提示词。")
        return prompt
        
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    generated_ids = prompt_ids.copy()

    print(f"正在基于提示词续写：{prompt}\n生成结果：", end="", flush=True)

    # 2. 开始逐字生成（贪心策略：每次只选概率最大的那个字）
    with torch.no_grad():
        for _ in range(max_length):
            # 生成当前序列长度的因果掩码
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
            
            # 模型前向传播，输出预测结果
            logits = model(input_ids, causal_mask)
            # 取出最后一个时间步（也就是最新位置）的预测分数
            next_token_logits = logits[:, -1, :]
            
            # 贪心采样的核心：直接取概率最大的那个字的索引 (argmax)
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # 如果遇到结束符（如果有的话）或者模型开始疯狂重复，可以提前停止
            # 这里简单判断一下是否生成了过长的重复片段（可选）
            
            # 将生成的新字加入序列，作为下一次的输入
            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            # 实时打印生成的字
            print(idx2char[next_token_id], end="", flush=True)
            
    print("\n\n生成完毕！")
    return ''.join([idx2char[i] for i in generated_ids])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt", help="训练好的模型路径")
    parser.add_argument("--prompt", default="从前有座山", help="续写的开头提示词")
    parser.add_argument("--max_length", type=int, default=100, help="最多生成多少个字")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载保存的模型和字典
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # 按照训练时的参数重新实例化模型
    model = TransformerLM(
        vocab_size=len(checkpoint['char2idx']),
        embed_dim=model_args['embed_dim'],
        hidden_dim=model_args['hidden_dim'],
        num_layers=model_args['num_layers'],
        dropout=model_args['dropout']
    ).to(device)
    
    # 载入训练好的权重
    model.load_state_dict(checkpoint['model_state'])
    
    greedy_generate(model, checkpoint, device, args.prompt, args.max_length)

if __name__ == "__main__":
    main()