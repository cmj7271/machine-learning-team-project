import pandas as pd
import numpy as np
import torch
import time
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from config import Config

# --- Resource Tracking Context Manager ---
class ResourceTracker:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.peak_memory = 0
        
    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak / (1024 * 1024) # Convert to MB
        tracemalloc.stop()
        
    def get_stats(self):
        return {
            "duration_sec": self.end_time - self.start_time,
            "peak_memory_mb": self.peak_memory
        }

# --- Data Loading ---
def load_data():
    print("📂 Loading Data...")
    try:
        goemotions = pd.read_csv(f'{Config.DATA_PATH}/clean_goemotions.csv').dropna(subset=['text'])
        reddit = pd.read_csv(f'{Config.DATA_PATH}/clean_reddit.csv').dropna(subset=['body'])
    except FileNotFoundError:
        raise FileNotFoundError("Data files not found in ./data folder.")

    # Sampling
    labeled = goemotions.sample(n=min(Config.LABELED_SIZE, len(goemotions)), random_state=42)
    unlabeled = reddit.sample(n=min(Config.UNLABELED_SIZE, len(reddit)), random_state=42)
    
    emotion_cols = [c for c in goemotions.columns if c not in ['text', 'word_count']]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        labeled['text'].values, labeled[emotion_cols].values, test_size=0.15, random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels, unlabeled['body'].values

# --- Training & Evaluation ---
def train_epoch(model, texts, labels, tokenizer, optimizer, criterion, device):
    model.train()
    indices = np.random.permutation(len(texts))
    total_loss = 0
    
    for i in range(0, len(indices), Config.BATCH_SIZE):
        batch_idx = indices[i:i+Config.BATCH_SIZE]
        inputs, masks = [], []
        batch_labels = []
        
        for idx in batch_idx:
            ids, mask = tokenizer.encode(texts[idx])
            inputs.append(ids)
            masks.append(mask)
            batch_labels.append(labels[idx])
            
        inp_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
        mask_tensor = torch.tensor(masks, dtype=torch.float).to(device)
        lbl_tensor = torch.tensor(batch_labels, dtype=torch.float).to(device)
        
        logits = model(inp_tensor, mask_tensor)
        loss = criterion(logits, lbl_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(indices) / Config.BATCH_SIZE)

def evaluate_metrics(model, texts, labels, tokenizer, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i in range(0, len(texts), Config.BATCH_SIZE):
            batch_texts = texts[i:i+Config.BATCH_SIZE]
            batch_lbls = labels[i:i+Config.BATCH_SIZE]
            
            inputs, masks = [], []
            for txt in batch_texts:
                ids, mask = tokenizer.encode(txt)
                inputs.append(ids)
                masks.append(mask)
                
            inp_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
            mask_tensor = torch.tensor(masks, dtype=torch.float).to(device)
            
            logits = model(inp_tensor, mask_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(batch_lbls)
            
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.array(all_labels), np.array(all_preds), average='macro', zero_division=0
    )
    return precision, recall, f1