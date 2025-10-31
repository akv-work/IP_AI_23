
import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


CLASS_NAMES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

# Базовая папка для сохранения всех результатов
base_dir = Path(r"C:\\Users\\User\\Desktop\\Studing-7sem\\OIIS\\lab_1")
models_dir = base_dir / 'models'
results_dir = base_dir / 'results'


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size=128, augment=True, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ] + train_transforms

    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc='train', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def eval_epoch(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='eval', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def plot_metrics(history):
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs = len(history['train_loss'])
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epochs+1), history['test_loss'], label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='train_acc')
    plt.plot(range(1, epochs+1), history['test_acc'], label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    out_path = results_dir / 'training_metrics.png'
    plt.savefig(out_path)
    print('Saved plot to', out_path)
    plt.close()


def predict_image(model, device, img_path):
    model.eval()
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(out.argmax(1).cpu().numpy()[0])

    plt.imshow(np.array(img))
    plt.title(f'Pred: {CLASS_NAMES[pred]} ({probs[pred]*100:.1f}%)')
    plt.axis('off')
    plt.show()
    return pred, probs


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    if args.mode == 'train':
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, augment=True)
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters())

        history = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}
        best_test_acc = 0.0

        for epoch in range(1, args.epochs+1):
            start = time.time()
            train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
            test_loss, test_acc = eval_epoch(model, device, test_loader, criterion)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            elapsed = time.time() - start
            print(f'Epoch {epoch}/{args.epochs}  Train loss {train_loss:.4f} acc {train_acc:.2f}%  |  Test loss {test_loss:.4f} acc {test_acc:.2f}%  ({elapsed:.1f}s)')

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({'model_state': model.state_dict(), 'args': vars(args)}, models_dir / 'model_best.pth')

        torch.save({'model_state': model.state_dict(), 'args': vars(args)}, models_dir / 'model.pth')
        plot_metrics(history)
        print('Training complete. Best test acc:', best_test_acc)

    elif args.mode == 'eval':
        _, test_loader = get_dataloaders(batch_size=args.batch_size, augment=False)
        model = SimpleCNN().to(device)
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = eval_epoch(model, device, test_loader, criterion)
        print(f'Test loss {test_loss:.4f}  Test acc {test_acc:.2f}%')

    elif args.mode == 'predict':
        model = SimpleCNN().to(device)
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        predict_image(model, device, args.img)

    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train','eval','predict'], default='train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-path', type=str, default=str(models_dir / 'model.pth'))
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    main(args)
