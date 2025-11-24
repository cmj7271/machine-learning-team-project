import torch
import json
import os
from config import Config
from models import HFTokenizerWrapper, SimpleMLP
from utils import load_data, train_epoch, evaluate_metrics, ResourceTracker

def main():
    NAME = "MLP_Baseline"
    save_dir = Config.get_dir(NAME)
    
    # Data & Tokenizer
    train_txt, val_txt, train_lbl, val_lbl, _ = load_data()
    tokenizer = HFTokenizerWrapper(max_length=Config.MAX_LEN)
    
    # Model
    model = SimpleMLP(tokenizer.vocab_size, Config.EMBED_DIM, Config.HIDDEN_DIM, Config.N_EMOTIONS).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR_MLP)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"🚀 Running {NAME}...")
    
    best_f1 = 0
    final_metrics = {}
    
    with ResourceTracker() as tracker:
        for epoch in range(Config.EPOCHS):
            loss = train_epoch(model, train_txt, train_lbl, tokenizer, optimizer, criterion, Config.DEVICE)
            p, r, f1 = evaluate_metrics(model, val_txt, val_lbl, tokenizer, Config.DEVICE)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                final_metrics = {'precision': p, 'recall': r, 'f1': f1}
                
    stats = tracker.get_stats()
    stats.update(final_metrics)
    
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✅ {NAME} Finished. Stats saved.")

if __name__ == "__main__":
    main()