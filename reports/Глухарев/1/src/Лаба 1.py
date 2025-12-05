import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ('airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck')


# === Улучшенная простая CNN ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 96 -> 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 48 -> 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 24 -> 12

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class STL10Dataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None):
        self.transform = transform
        with open(data_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
            data = data.reshape(-1, 3, 96, 96)
            self.data = np.transpose(data, (0, 2, 3, 1))
        with open(labels_file, 'rb') as f:
            self.labels = np.fromfile(f, dtype=np.uint8) - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label



def get_dataloaders(batch_size=64, num_workers=2):
    transform_train = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # аугментация
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
    ])

    train_set = STL10Dataset(
        data_file='./data/train_X.bin',
        labels_file='./data/train_y.bin',
        transform=transform_train
    )
    test_set = STL10Dataset(
        data_file='./data/test_X.bin',
        labels_file='./data/test_y.bin',
        transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader



def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(dataloader, desc='Train batches', leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, device, dataloader, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / len(dataloader.dataset) if criterion is not None else None
    accuracy = correct / total
    return avg_loss, accuracy


def plot_losses(train_losses, val_losses, out_path=None):
    epochs = np.arange(1, len(train_losses)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.grid(True)
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f'Saved loss plot to {out_path}')
    plt.show()


def show_predictions(model, device, dataloader, num_images=8):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15,6))

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                img = images[i].cpu().permute(1,2,0).numpy()
                true_label = CLASS_NAMES[targets[i].item()]
                pred_label = CLASS_NAMES[preds[i].item()]

                plt.subplot(2, num_images//2, images_shown+1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
                images_shown += 1
            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32 )
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save-model', type=str, default='stl10_cnn.pth')
    parser.add_argument('--predict', type=str, default=None, help='путь к изображению для предсказания')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []
    val_losses = []
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'  Train loss: {train_loss:.4f}')
        print(f'  Val loss:   {val_loss:.4f}, Val acc: {val_acc*100:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_model)
            print(f'  Saved best model (acc={best_acc*100:.2f}%) to {args.save_model}')

        scheduler.step()

    plot_losses(train_losses, val_losses, out_path='loss_curve.png')

    # показать
    show_predictions(model, device, test_loader, num_images=8)


if __name__ == '__main__':
    main()