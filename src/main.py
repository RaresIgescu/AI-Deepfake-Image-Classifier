import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter

# --- 1. Dataset pentru train/val ---
class DeepfakeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, f"{row.image_id}.png")
        img   = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.long)
        return img, label

# --- 2. Model simplu from scratch ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*28*28,256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- 3. Dataset pentru test (fără label) ---
class DeepfakeTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        path   = os.path.join(self.img_dir, f"{img_id}.png")
        img    = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img_id, img

# --- 4. Funcții de antrenare și validare ---
def train_epoch_verbose(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = n_samples = 0

    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds     = outputs.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()
        batch_loss= loss.item()

        # distribuția etichetelor
        dist      = Counter(labels.cpu().numpy())
        dist_full = {cls: dist.get(cls,0) for cls in range(model.classifier[-1].out_features)}

        print(f"[Batch {batch_idx:03d}/{len(loader)}] "
              f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f} | Label dist: {dist_full}")

        bs = labels.size(0)
        total_loss    += batch_loss * bs
        total_correct += (preds == labels).sum().item()
        n_samples     += bs

    epoch_loss = total_loss / n_samples
    epoch_acc  = total_correct / n_samples
    print(f"→ Epocă terminată: Loss mediu {epoch_loss:.4f}, Acc medie {epoch_acc:.4f}\n")
    return epoch_loss, epoch_acc

def eval_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs       = model(imgs)
            loss          = criterion(outputs, labels)
            total_loss   += loss.item() * imgs.size(0)
            total_correct+= (outputs.argmax(1)==labels).sum().item()
    n = len(loader.dataset)
    return total_loss/n, total_correct/n

# --- 5. Funcția main ---
def main():
    # Detectare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    num_epochs = 0  # setează >0 pentru a antrena, 0 pentru skip training

    # Transformări comune
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Încarcă metadata
    train_df = pd.read_csv("../data/train.csv")
    val_df   = pd.read_csv("../data/validation.csv")
    test_df  = pd.read_csv("../data/test.csv")

    # DataLoader pentru train/val
    train_ds       = DeepfakeDataset(train_df,   "../data/train",      transform)
    val_ds         = DeepfakeDataset(val_df,     "../data/validation", transform)
    train_loader   = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader     = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model     = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ckpt = "best_model.pth"
    # Dacă num_epochs==0 => skip training și doar încărcăm modelul
    if num_epochs == 0:
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"✔️  Loaded saved model (skip training, num_epochs=0)")
        else:
            raise RuntimeError(f"No checkpoint at {ckpt} but num_epochs=0")
    else:
        # Dacă dorim antrenare
        best_val_acc = 0.0
        # (opțional) start de la checkpoint vechi
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"🔄 Loaded {ckpt} for continued training")
        for epoch in range(1, num_epochs+1):
            train_epoch_verbose(train_loader, model, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(val_loader, model, criterion, device)
            print(f"Epoch {epoch:02d} | Val loss {val_loss:.4f}, Val acc {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt)
                print(f"↪️  Saved best model (val_acc={val_acc:.4f})\n")

    # Inferență și generare submission.csv
    test_ds     = DeepfakeTestDataset(test_df, "../data/test", transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)

    model.eval()
    preds, ids = [], []
    with torch.no_grad():
        for img_ids, imgs in test_loader:
            imgs   = imgs.to(device)
            outs   = model(imgs)
            labels = outs.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(labels)
            ids.extend(img_ids)

    sub = pd.DataFrame({"image_id": ids, "label": preds})
    sub.to_csv("submission.csv", index=False)
    print(f"✔️  Wrote submission.csv ({len(sub)} rows)")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
