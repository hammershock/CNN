from ray import tune
from ray.train import report
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from net import Net

# 定义超参数空间
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
    "momentum": tune.uniform(0.8, 0.99),
    "weight_decay": tune.loguniform(1e-6, 1e-1)
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def test_accuracy(loader, model, device=torch.device("cpu")):
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, running_loss / len(loader)


def train_mnist(config):
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True)

    for epoch in range(10):  # 使用较少的epoch以加快调试速度
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        validation_accuracy, val_loss = test_accuracy(test_loader, net)
        # 报告结果给Ray Tune
        report({"accuracy": validation_accuracy, "loss": val_loss, "training_iteration": epoch})  # loss=loss.item(),


if __name__ == "__main__":
    # 测试集加载器
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # 启动Ray Tune
    scheduler = ASHAScheduler(
        max_t=10,  # 最大迭代次数
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    analysis = tune.run(
        train_mnist,
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=10,
        metric="accuracy",
        mode="max",
        scheduler=scheduler,
        progress_reporter=reporter)

    dfs = analysis.trial_dataframes

    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
