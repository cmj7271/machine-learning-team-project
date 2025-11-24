import torch
import numpy as np
import json
import os
from config import Config
from models import HFTokenizerWrapper, KDA_Classifier
from utils import load_data, train_epoch, evaluate_metrics, ResourceTracker

def pseudo_label_step(model, unlabeled_texts, tokenizer, device, threshold=0.85):
    model.eval()
    new_texts, new_labels = [], []
    
    # 샘플링하여 일부만 사용 (속도 최적화)
    sample_indices = np.random.choice(len(unlabeled_texts), size=min(1000, len(unlabeled_texts)), replace=False)
    subset_texts = unlabeled_texts[sample_indices]
    
    with torch.no_grad():
        for txt in subset_texts:
            ids, mask = tokenizer.encode(txt)
            inp = torch.tensor([ids]).to(device)
            msk = torch.tensor([mask]).to(device)
            logits = model(inp, msk)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            
            high_conf = (probs > threshold).astype(int)
            if high_conf.sum() > 0:
                new_texts.append(txt)
                new_labels.append(high_conf)
    return new_texts, new_labels

def main():
    NAME = "KDA_SemiSupervised"
    save_dir = Config.get_dir(NAME)
    
    train_txt, val_txt, train_lbl, val_lbl, unlabeled_txt = load_data()
    tokenizer = HFTokenizerWrapper(max_length=Config.MAX_LEN)
    
    model = KDA_Classifier(tokenizer.vocab_size, Config.EMBED_DIM, Config.HIDDEN_DIM, Config.N_EMOTIONS).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR_LSTM)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"🚀 Running {NAME} (with Pseudo-labeling)...")
    best_f1 = 0
    final_metrics = {}
    
    with ResourceTracker() as tracker:
        for epoch in range(Config.EPOCHS):
            # Regular Training
            loss = train_epoch(model, train_txt, train_lbl, tokenizer, optimizer, criterion, Config.DEVICE)
            
            # Pseudo Labeling (after epoch 2)
            if epoch >= 2:
                p_txt, p_lbl = pseudo_label_step(model, unlabeled_txt, tokenizer, Config.DEVICE)
                if p_txt:
                    print(f"  -> Pseudo-labeled {len(p_txt)} samples.")
                    # Augment & Train once
                    aug_txt = np.concatenate([train_txt, p_txt])
                    aug_lbl = np.vstack([train_lbl, p_lbl])
                    _ = train_epoch(model, aug_txt, aug_lbl, tokenizer, optimizer, criterion, Config.DEVICE)
            
            p, r, f1 = evaluate_metrics(model, val_txt, val_lbl, tokenizer, Config.DEVICE)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                final_metrics = {'precision': p, 'recall': r, 'f1': f1}
                
    stats = tracker.get_stats()
    stats.update(final_metrics)
    
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✅ {NAME} Finished.")

if __name__ == "__main__":
    main()