import os
import time
import copy
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

# ======================
# Global helper functions
# ======================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def repeat_gray(x):
    if x.shape[0] == 1:
        return x.repeat(3,1,1)
    return x

def get_model(num_classes=10, pretrained=True, repeat_gray=True):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    if not repeat_gray:
        old_conv = model.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:,0:1,:,:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        model.conv1 = new_conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, verbose=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if verbose and i % 50 == 0:  # каждые 50 батчей
            print(f'Batch {i}/{len(loader)} | Loss: {running_loss/total:.4f} | Acc: {correct/total:.4f}', flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, verbose=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if verbose and i % 50 == 0:
                print(f'Validation Batch {i}/{len(loader)} | Loss: {running_loss/total:.4f} | Acc: {correct/total:.4f}', flush=True)

    return running_loss / total, correct / total



def load_image_from_path_or_url(path_or_url, resize=224, repeat_gray=True):
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        resp = requests.get(path_or_url)
        img = Image.open(BytesIO(resp.content)).convert('L')
    else:
        img = Image.open(path_or_url).convert('L')
    img_for_model = img.resize((resize, resize))
    img_tensor = transforms.ToTensor()(img_for_model)
    if repeat_gray:
        img_tensor = img_tensor.repeat(3,1,1)
    img_tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_tensor)
    return img, img_tensor.unsqueeze(0)

def predict_and_show(model, path_or_url):
    model.eval()
    img, tensor = load_image_from_path_or_url(path_or_url)
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    print(f'Predicted class: {pred}  (prob={probs[pred]:.4f})')
    display_img = img.resize((200,200)).convert('RGB')
    display_img.show()

# ======================
# Main script
# ======================
# ======================
# Main script modification
# ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=str, help="Path to image for prediction, default 'digit.png'",
                        default="digit.png")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = './data'
    BATCH_SIZE = 128
    NUM_EPOCHS = 8
    IMAGE_SIZE = 64
    NUM_CLASSES = 10
    MODEL_SAVE = 'resnet34_mnist_adadelta_best.pth'
    PLOT_SAVE = 'training_curves.png'
    PRED_SAVE = 'prediction.png'
    USE_PRETRAINED = True
    REPEAT_GRAY_TO_3 = True

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(repeat_gray),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(repeat_gray),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(NUM_CLASSES, pretrained=USE_PRETRAINED, repeat_gray=REPEAT_GRAY_TO_3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ======================
    # Training loop
    # ======================
    for epoch in range(NUM_EPOCHS):
        print(f'\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===', flush=True)
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE)

        scheduler.step()
        t1 = time.time()
        print(f'Epoch {epoch + 1} finished | train_loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} acc={val_acc:.4f} | time={(t1 - t0):.1f}s', flush=True)

    # Load best weights and save final model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE.replace('.pth', '_final.pth'))
    print(f"Saved best model as {MODEL_SAVE} and final model as {MODEL_SAVE.replace('.pth', '_final.pth')}")

    # ======================
    # Plot training curves
    # ======================
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='train_loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_accs, label='train_acc')
    plt.plot(range(1, NUM_EPOCHS + 1), val_accs, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE)
    print(f'Saved training curves to {PLOT_SAVE}')

    # ======================
    # Predict on digit.png and save prediction image
    # ======================
    model.eval()
    img_path = args.predict
    img, tensor = load_image_from_path_or_url(img_path)
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    print(f'Predicted class: {pred}  (prob={probs[pred]:.4f})')

    # Save image with prediction
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {pred}")
    plt.axis('off')
    plt.savefig(PRED_SAVE)
    print(f"Saved prediction image as {PRED_SAVE}")
