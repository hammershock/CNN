import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import logging

from net import Net, NetWithBatchNorm, NetWithLayerNorm
from tune import test_accuracy


def plot(*args):
    epochs = range(1, len(args[0]) + 1)
    lst1, lst2, lst3 = args
    plt.plot(epochs, lst1, label='No Normalization')
    plt.plot(epochs, lst2, label='Batch Normalization')
    plt.plot(epochs, lst3, label='Layer Normalization')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # 显示图像
    plt.show()


if __name__ == "__main__":
    accuracies = []
    accuracies_bn = []
    accuracies_ln = []

    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # 数据集的转换器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    for net, lst in zip([Net(), NetWithBatchNorm(), NetWithLayerNorm()], [accuracies, accuracies_bn, accuracies_ln]):
        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001449, momentum=0.90358, weight_decay=0.000111403)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        num_epochs = 10

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            net.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    torch.save(net.state_dict(), f"checkpoints/ckpt{epoch}.pth")
                    logging.info(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            net.eval()
            validation_accuracy, validation_loss = test_accuracy(testloader, net, device=device)
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {validation_accuracy:.2f}%, Validatioin Loss: {validation_loss: .3f}')
            lst.append(validation_accuracy)

    plot(accuracies, accuracies_bn, accuracies_ln)
    print('Finished Training')


