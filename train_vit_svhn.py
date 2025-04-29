import torch
import torch.nn as nn
import torch.optim as optim
import timm
import umap
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def main():
    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 2. Load data
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset  = datasets.SVHN(root='./data', split='test',  download=True, transform=transform)

    train_dataset = Subset(train_dataset, list(range(3000)))
    test_dataset  = Subset(test_dataset, list(range(2000)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 3. Model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
    model.to(device)

    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False

    # 4. Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training
    print("Starting training...")
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

    print("Training complete.")

    # 6. Extract Features
    print("Extracting features...")
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            feats = model.forward_features(images)
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features).numpy()[:, 0, :]  # <-- pick only the CLS token
    labels = torch.cat(labels).numpy()
    print(f"Features shape: {features.shape}")

    # 7. UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)
    print(f"Embedding shape: {embedding.shape}")

    # 8. Plot
    print("Plotting...")
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title("UMAP Projection of ViT Features on SVHN Digits")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Only needed on Windows
    main()
