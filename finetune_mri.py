import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DATA_ROOT = r"D:\data\brats\kaggle_3m"
CSV_PATH = os.path.join(DATA_ROOT, "data.csv")
SSL_ENCODER_PATH = r"D:\MajorProject\ssl_mri_encoder.pth"
SAVE_MODEL_PATH = r"D:\MajorProject\finetuned_mri_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4

class MRIDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            patient_id = row["Patient"]
            label = row["death01"]
            if pd.isna(label): continue
            for folder in os.listdir(root_dir):
                if folder.startswith(patient_id):
                    patient_folder = os.path.join(root_dir, folder)
                    for img in os.listdir(patient_folder):
                        if img.endswith(".tif"):
                            self.samples.append((os.path.join(patient_folder, img), int(label)))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

class MRISSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.classifier(feats).squeeze()

def train_mri_model():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    dataset = MRIDataset(CSV_PATH, DATA_ROOT, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MRISSLModel().to(DEVICE)
    ssl_state = torch.load(SSL_ENCODER_PATH, map_location=DEVICE)
    model.encoder.load_state_dict(ssl_state, strict=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"✅ MRI fine-tuned model saved at {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train_mri_model()