import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm  # used to create progress bars for for-loops


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim,
        criterion: nn,
        device: torch.device,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def eval(self, model, checkpoint_name: str = "best_model") -> None:
        train_loss = []
        test_loss = []
        min_loss = float("inf")
        train_acc = []
        test_acc = []
        for epoch in range(self.epochs):
            epoch_train_loss, epoch_train_acc = self._train(model=model)
            train_loss.append(epoch_train_loss)
            train_acc.append(epoch_train_acc)
            epoch_test_loss, epoch_test_acc = self._test(model=model)
            if epoch_test_loss < min_loss:
                min_loss = epoch_test_loss
                torch.save(model, f"{checkpoint_name}.pth")
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)
            print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss} | Test Loss: {epoch_test_loss}")
            # print(f"Epoch: {epoch} | Train Accuracy: {epoch_train_acc} | Test Accuracy: {epoch_test_acc}")
        self._plot_metric(train=train_loss, test=test_loss, metric="loss")
        # self._plot_metric(train=train_acc, test=test_acc, metric="accuracy")

    def check_accuracy(self, model) -> None:
        if self.val_loader is None:
            print("No validation loader was provided. Cannot check accuracy.")
            return
        else:
            print("Checking accuracy on validation set")
            self._check_accuracy(model=model)

    def _train(self, model) -> float:
        model.train()
        train_loss = 0
        train_accuracy = 0
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            # Get data to cuda if possible
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            # forward
            output = model(data)
            loss = self.criterion(output, targets.unsqueeze(1))
            # backward
            loss.backward()
            # calc accuracy
            accuracy = (output.argmax(dim=1) == targets).float().mean()
            # optimizer step
            self.optimizer.step()
            # update train loss
            train_loss += loss.item()
            train_accuracy += accuracy.item()
        return (train_loss / (batch_idx + 1)), (train_accuracy / (batch_idx + 1))

    def _test(self, model) -> float:
        model.eval()
        test_loss = 0
        train_accuracy = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.test_loader)):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                # forward
                output = model(data)
                loss = self.criterion(output, targets.unsqueeze(1))
                # calc accuracy
                accuracy = (output.argmax(dim=1) == targets).float().mean()
                # update test loss
                test_loss += loss.item()
                train_accuracy += accuracy.item()
        return (test_loss / (batch_idx + 1)), (train_accuracy / (batch_idx + 1))

    def _check_accuracy(self, model) -> None:
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(self.val_loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct) / float(num_samples) * 100:.2f}%"
            )

    def _plot_metric(self, train, test, metric: str) -> None:
        plt.plot(train, "-b", label=f"train {metric}")
        plt.plot(test, label=f"test {metric}")
        plt.title(f"{metric} vs epoch")
        plt.xlabel("epoch")
        plt.ylabel(f"{metric}")
        plt.grid(True)
        plt.legend()
        plt.show()
