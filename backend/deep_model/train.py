import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SpillNet
from dataset import SpillDataset

# -----------------------------------
# Device
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# Dataset path (CORRECT)
# -----------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data")

dataset = SpillDataset(DATASET_PATH)
print(f"✅ Loaded {len(dataset)} samples")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -----------------------------------
# Model
# -----------------------------------
model = SpillNet().to(device)

# -----------------------------------
# Losses
# -----------------------------------
loss_age = nn.MSELoss()
loss_thick = nn.MSELoss()
loss_risk = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------------
# TRAINING LOOP (THIS WAS MISSING)
# -----------------------------------
EPOCHS = 3   # CPU-friendly

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img, age, thick, risk in loader:
        img = img.to(device)
        age = age.float().to(device)
        thick = thick.float().to(device)
        risk = risk.to(device)

        pred_age, pred_thick, pred_risk = model(img)

        loss = (
            loss_age(pred_age.squeeze(), age) +
            loss_thick(pred_thick.squeeze(), thick) +
            loss_risk(pred_risk, risk)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -----------------------------------
# SAVE MODEL (VERY IMPORTANT)
# -----------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "spillnet.pth")
torch.save(model.state_dict(), MODEL_PATH)

print(f"✅ Deep model trained and saved at: {MODEL_PATH}")
