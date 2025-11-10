import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import ColorizationDataset
from model import ColorizationNet

def train_model(data_dir, epochs=10, batch_size=16, lr=1e-3, device="cuda"):
    dataset = ColorizationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ColorizationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for L, ab in dataloader:
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/colorization_model.pth")
    print("✅ Model saved at models/colorization_model.pth")
