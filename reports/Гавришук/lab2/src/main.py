import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

batch_size = 64
learning_rate = 0.01
momentum = 0.9
num_epochs = 10

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])


def create_modified_mobilenet_v3(num_classes=100):
    model = models.mobilenet_v3_large(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Финальная точность на тестовой выборке: {accuracy:.2f}%')
    return accuracy


def visualize_prediction(model, test_dataset, image_path=None):
    if image_path is None:
        random_index = np.random.randint(0, len(test_dataset))
        image, true_label = test_dataset[random_index]
        image = image.unsqueeze(0).to(device)
    else:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image = transform_test(image).unsqueeze(0).to(device)
        true_label = None

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = test_dataset.classes

    image_to_show = image.squeeze(0).cpu()
    image_to_show = image_to_show.permute(1, 2, 0)

    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])
    image_to_show = image_to_show * std + mean
    image_to_show = torch.clamp(image_to_show, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_to_show)

    if true_label is not None:
        plt.title(f'Предсказание: {classes[predicted.item()]} ({confidence.item():.2f})\n'
                  f'Истинный класс: {classes[true_label]}')
    else:
        plt.title(f'Предсказание: {classes[predicted.item()]} (уверенность: {confidence.item():.2f})')

    plt.axis('off')
    plt.show()

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("Топ-5 предсказаний:")
    for i in range(top5_prob.size(1)):
        print(f"{classes[top5_catid[0][i].item()]}: {top5_prob[0][i].item():.4f}")


def main():
    print("Загрузка данных CIFAR-100...")

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 для Windows
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")
    print(f"Количество классов: {len(train_dataset.classes)}")

    print("Создание модели MobileNet v3...")
    model = create_modified_mobilenet_v3(num_classes=100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum
    )

    print("Начало обучения...")
    train_losses, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
    plt.title('Изменение ошибки во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', linewidth=2)
    plt.title('Точность на тестовой выборке')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    final_accuracy = evaluate_model(model, test_loader)

    print("\nВизуализация работы сети на случайном изображении:")
    visualize_prediction(model, test_dataset)

    torch.save(model.state_dict(), 'mobilenet_v3_cifar100.pth')
    print("Модель сохранена как 'mobilenet_v3_cifar100.pth'")

    print("\nДополнительная визуализация на 3 случайных изображениях:")
    for i in range(3):
        visualize_prediction(model, test_dataset)


if __name__ == '__main__':
    main()