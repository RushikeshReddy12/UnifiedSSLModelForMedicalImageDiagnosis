import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_DIR = r"D:\data\luna16\2d_images"
CSV_PATH = r"D:\data\luna16\lung_stats.csv"
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CTDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(CSV_PATH)
        self.df["label"] = (self.df["lung_mean_hu"] < -700).astype(int)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, row["img_id"])
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

class CTModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512,1)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        return self.classifier(x)

def train_ct_model():
    model = CTModel().to(DEVICE)
    ssl_weights = torch.load("ssl_ct_encoder.pth", map_location=DEVICE)
    model.encoder.load_state_dict(ssl_weights, strict=False)
    for param in model.encoder.parameters():
        param.requires_grad = False
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
    loader = DataLoader(CTDataset(), batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "finetuned_ct_model.pth")
    print("✅ CT model fine-tuned and saved")

if __name__ == "__main__":
    train_ct_model()