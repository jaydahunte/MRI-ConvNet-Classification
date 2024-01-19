import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pickle
import numpy as np
from trainer import Trainer

# Hyper-parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pickle.load(open("data_tensor.pkl", "rb"))


def split_data(data, split_pct):
    size = int(len(data) * (1 - split_pct))
    data, test_data = torch.utils.data.random_split(data, [size, len(data) - size])
    return data, test_data


train_data, test_data = split_data(data, split_pct=0.3)
test_data, val_data = split_data(test_data, split_pct=0.33)

TRAIN_LOADER = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
VAL_LOADER = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

CRITERION = nn.BCEWithLogitsLoss()


def weighted_accuracy(confusion):
    # input: confusion is the confusion matrix
    # output: acc is the weighted classification accuracy
    M = confusion.copy().astype("float32")
    for k in range(0, M.shape[0]):
        M[k] /= M[k].sum() + 1e-8
    acc = M.diagonal().sum() / M.sum()
    return acc


def fit_model(features: np.ndarray, target: np.ndarray, scaler, model) -> None:
    X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(features, target, test_size=0.3, random_state=17)
    X_TEST, X_VAL, y_TEST, y_VAL = train_test_split(X_TEST, y_TEST, test_size=0.33, random_state=17)

    scaler.fit(X_TRAIN)  # think about why fit to X_train, not X ?
    X_train = scaler.transform(X_TRAIN)
    X_val = scaler.transform(X_VAL)
    X_test = scaler.transform(X_TEST)

    model.fit(X_train, y_TRAIN)
    y_val_pred = model.predict(X_val)
    confusion_val = confusion_matrix(y_VAL, y_val_pred)
    acc_val = weighted_accuracy(confusion_val)
    y_test_pred = model.predict(X_test)
    confusion_test = confusion_matrix(y_TEST, y_test_pred)
    acc_test = weighted_accuracy(confusion_test)
    print("classification accuracy on validation set is ", acc_val)
    print("classification accuracy on test set is ", acc_test)


def eval_convnet(model, checkpoint_name: str = "best_model") -> None:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        train_loader=TRAIN_LOADER,
        test_loader=TEST_LOADER,
        val_loader=VAL_LOADER,
        optimizer=optimizer,
        criterion=CRITERION,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    trainer.eval(model=model, checkpoint_name=checkpoint_name)

    # inference
    load_eval_convnet(checkpoint_name=checkpoint_name)


def load_eval_convnet(checkpoint_name: str = "best_model") -> None:
    model = torch.load(f"{checkpoint_name}.pth")
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(VAL_LOADER):
            # Get data to cuda if possible
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            # forward
            output = model(data)
            if (nn.Sigmoid()(output) > 0.5 and targets == 1) or (nn.Sigmoid()(output) <= 0.5 and targets == 0):
                correct += 1

    val_acc = correct / len(VAL_LOADER)

    print(f"Validation accuracy: {val_acc}")
