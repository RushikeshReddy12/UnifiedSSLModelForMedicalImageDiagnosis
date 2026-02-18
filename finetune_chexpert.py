import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DATA_ROOT = r"D:\data\cheXpert"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid.csv")
SSL_ENCODER_PATH = "ssl_chexpert_encoder.pth"
BATCH_SIZE = 8
EPOCHS = 2
LR = 3e-4
IMG_SIZE = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "finetuned_chexpert_model.pth"
DISEASES = ["No Finding", "Lung Opacity", "Lung Lesion",
            "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Fracture"]

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, ignore_uncertain=True):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.sample(n=min(500, len(self.df)), random_state=42).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.ignore_uncertain = ignore_uncertain
        if ignore_uncertain:
            self.df = self.df.loc[(self.df[DISEASES] != -1).any(axis=1)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.df.iloc[idx]["Path"].replace("CheXpert-v1.0-small/", ""))
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)

        labels = self.df.iloc[idx][DISEASES].values.astype(float)
        labels = [0 if (l == -1 or l != l) else l for l in labels]
        labels = [min(max(l, 0.0), 1.0) for l in labels]
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class CheXpertClassifier(nn.Module):
    def __init__(self, ssl_encoder_path, num_classes=len(DISEASES)):
        super().__init__()
        base_encoder = models.resnet18(weights=None)
        base_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        self.encoder.load_state_dict(torch.load(ssl_encoder_path, map_location=DEVICE))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.classifier(features)

def train():
    train_dataset = CheXpertDataset(TRAIN_CSV, DATA_ROOT, transform)
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_ROOT, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = CheXpertClassifier(SSL_ENCODER_PATH).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Fine-tuned model saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train()