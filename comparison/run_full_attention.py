import torch
import torch.nn.utils as nn_utils
import json
import os
import sys

from models import HFTokenizerWrapper, FullAttention_Classifier
from utils import load_data, evaluate_metrics, ResourceTracker
from config import Config

def train_epoch(model, texts, labels, tokenizer, optimizer, criterion, device, grad_clip=1.0):
    """Gradient Clipping이 추가된 학습 함수"""
    import numpy as np
    
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
        
        # Gradient Clipping
        # if grad_clip is not None:
        #     nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(indices) / Config.BATCH_SIZE)

def main():
    NAME = "FullAttention"
    save_dir = Config.get_dir(NAME)
    
    train_txt, val_txt, train_lbl, val_lbl, _ = load_data()
    tokenizer = HFTokenizerWrapper(max_length=Config.MAX_LEN)
    
    # Full Attention 모델
    model = FullAttention_Classifier(
        tokenizer.vocab_size, 
        Config.EMBED_DIM, 
        Config.HIDDEN_DIM, 
        Config.N_EMOTIONS,
        dropout=0.5
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # # Learning Rate Scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=2
    # )
    
    print(f"🚀 Running {NAME} (O(n²) Complexity)...")
    print(f"  - Full Scaled Dot-Product Attention")
    print(f"  - Learning Rate: 5e-4")
    print(f"  - Dropout: 0.5")
    
    best_f1 = 0
    final_metrics = {}
    
    with ResourceTracker() as tracker:
        for epoch in range(Config.EPOCHS):
            loss = train_epoch(model, train_txt, train_lbl, tokenizer, 
                             optimizer, criterion, Config.DEVICE, grad_clip=1.0)
            
            p, r, f1 = evaluate_metrics(model, val_txt, val_lbl, tokenizer, Config.DEVICE)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, F1={f1:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
            
            # scheduler.step(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                final_metrics = {'precision': p, 'recall': r, 'f1': f1}
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                
    stats = tracker.get_stats()
    stats.update(final_metrics)
    
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✅ {NAME} Finished. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()