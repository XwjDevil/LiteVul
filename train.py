import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 只用第1张卡
import copy
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from regvd_model import SimpleEdgeAwareUniNet
import numpy as np

# 固定随机种子，确保实验可复现
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用非确定性优化

# 加载数据
def load_data(pkl_path, batch_size, shuffle=True, device="cuda"):
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)
    use_cuda = str(device).startswith("cuda")
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  
        pin_memory=use_cuda,
        persistent_workers=False
    )

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return acc, prec, rec, f1

# 训练过程
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    y_true_all, y_pred_all = [], []
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
        preds = out.argmax(dim=1).detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        y_true_all.extend(labels.tolist())
        y_pred_all.extend(preds.tolist())
    acc, prec, rec, f1 = get_metrics(y_true_all, y_pred_all)
    avg_loss = total_loss / len(y_true_all)
    return avg_loss, acc, prec, rec, f1

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        y_true_all.extend(labels.tolist())
        y_pred_all.extend(preds.tolist())
    acc, prec, rec, f1 = get_metrics(y_true_all, y_pred_all)
    return acc, prec, rec, f1

# 主训练过程
def main():
    set_seed(41525)

    # 超参
    in_dim = 768
    hidden_dim = 256
    num_classes = 2
    num_layers = 4
    heads = 4
    batch_size = 32
    base_lr = 1e-4
    weight_decay = 1e-2
    epochs = 100
    early_stop_patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/mnt/data/NewReGVD/Devign/Devign_PPMI/pkl"
    train_loader = load_data(f"{data_dir}/train.pkl", batch_size, shuffle=True, device=str(device))
    val_loader   = load_data(f"{data_dir}/val.pkl", batch_size, shuffle=False, device=str(device))
    test_loader  = load_data(f"{data_dir}/test.pkl", batch_size, shuffle=False, device=str(device))

    model = SimpleEdgeAwareUniNet(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        heads=heads,
        use_degree_gate=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    header = f"{'Epoch':<7}{'LR':>10}{'TrLoss':>10}{'TrAcc':>10}{'TrF1':>10} | {'VaAcc':>10}{'VaF1':>10}"
    print(header)

    for epoch in range(1, epochs + 1):
        train_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, device
        )
        va_acc, va_prec, va_rec, va_f1 = eval_model(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:<7}{current_lr:>10.2e}{train_loss:>10.4f}{tr_acc:>10.4f}{tr_f1:>10.4f} | {va_acc:>10.4f}{va_f1:>10.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        te_acc, te_prec, te_rec, te_f1 = eval_model(model, test_loader, device)
        print("=" * 60)
        print(f"Test@BestValF1 (epoch {best_epoch})")
        print(f"  Acc       : {te_acc:.4f}")
        print(f"  Precision : {te_prec:.4f}")
        print(f"  Recall    : {te_rec:.4f}")
        print(f"  F1        : {te_f1:.4f}")
        #torch.save(model.state_dict(), os.path.join(data_dir, "simple_uninet_best.pt"))
        ablation_tag = "full"   # 或 "gat" / "gcn" / "nojk"
        save_path = os.path.join(data_dir, f"simple_uninet_{ablation_tag}_best.pt")
        torch.save(model.state_dict(), save_path)
    else:
        print("No best state captured. Check training/validation pipeline.")

if __name__ == "__main__":
    main()













































