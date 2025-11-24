import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# ==========================================
# Tokenizer (공통 사용)
# ==========================================
class HFTokenizerWrapper:
    def __init__(self, model_name='bert-base-uncased', max_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
        
    def fit(self, texts):
        pass
        
    def encode(self, text):
        text = str(text) if text is not None else ""
        encoded = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0).tolist(), encoded['attention_mask'].squeeze(0).tolist()

# ==========================================
# 1. Simple MLP (Baseline)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_emotions, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_emotions)
        )
        
    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            embedded = embedded * mask
            sum_embed = embedded.sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = sum_embed / count
        else:
            pooled = embedded.mean(dim=1)
        return self.fc(pooled)

# ==========================================
# 2. Full Attention (O(n²) complexity)
# ==========================================
class FullAttention_Classifier(nn.Module):
    """전통적인 Scaled Dot-Product Attention (Baseline for comparison)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_emotions, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, 
                           dropout=dropout, batch_first=True)
        
        # 🔥 Full Scaled Dot-Product Attention (O(n²))
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.scale = (hidden_dim * 2) ** -0.5
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_emotions)
        )
        
    def forward(self, x, mask=None):
        # Embedding + LSTM
        x = self.embedding(x)  # [batch, seq_len, embed]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # 🔥 Full Attention: Q, K, V transformation
        Q = self.query(lstm_out)  # [batch, seq_len, hidden*2]
        K = self.key(lstm_out)    # [batch, seq_len, hidden*2]
        V = self.value(lstm_out)  # [batch, seq_len, hidden*2]
        
        # Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d)V
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [batch, seq_len, seq_len] ← O(n²)
        
        # Masking
        if mask is not None:
            # mask: [batch, seq_len] → [batch, 1, seq_len]
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]
        context = torch.bmm(attn_weights, V)  # [batch, seq_len, hidden*2]
        
        # Global pooling
        if mask is not None:
            mask_expanded = mask.transpose(1, 2)  # [batch, seq_len, 1]
            context = context * mask_expanded
            pooled = context.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = context.mean(dim=1)
        
        return self.classifier(pooled)

# ==========================================
# 3. Kimi Linear Attention (O(n) complexity)
# ==========================================
class KDA_Classifier(nn.Module):
    """Bi-LSTM + Gated Linear Attention (Kimi-style)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_emotions, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, 
                           dropout=dropout, batch_first=True)
        
        # 🔥 Linear Attention: Gate + Linear Attention (O(n))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Sigmoid()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # ← 각 position별로 하나의 score (not pairwise)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_emotions)
        )
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # 🔥 Gated mechanism
        gate_scores = self.gate(lstm_out)  # [batch, seq_len, hidden*2]
        gated_lstm = lstm_out * gate_scores
        
        # 🔥 Linear Attention: 각 position의 importance만 계산 (O(n))
        attn_scores = self.attention(gated_lstm).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), gated_lstm).squeeze(1)  # [batch, hidden*2]
        
        return self.classifier(context)

# ==========================================
# 복잡도 비교
# ==========================================
"""
모델별 Attention 복잡도:

1. MLP: 
   - No attention
   - Complexity: O(1)

2. Full Attention:
   - QK^T: [batch, seq_len, hidden] × [batch, hidden, seq_len] = [batch, seq_len, seq_len]
   - Complexity: O(n²d) where n=seq_len, d=hidden_dim
   - Memory: O(n²)

3. KDA (Linear Attention):
   - Attention score per position: [batch, seq_len] → [batch, 1]
   - Complexity: O(nd)
   - Memory: O(n)

성능 목표:
- Full Attention: 최고 성능, 하지만 O(n²) 복잡도
- KDA: Full과 비슷한 성능, O(n) 복잡도 (효율성 증명!)
"""