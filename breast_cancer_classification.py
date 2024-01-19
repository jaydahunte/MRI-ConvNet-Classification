import torch
import torch.nn as nn  # neural network modules
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pickle

from resnet import ResNet18, ResNet50, ResNet101, ResNet152
from vgg import VGG
from model_utils import eval_convnet, fit_model

torch.manual_seed(17)  # computers a (pseudo) random, so specifying a seed allows for reproducibility

imgs = pickle.load(open("imgs.pkl", "rb"))
target = pickle.load(open("labels.pkl", "rb"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18_transfer_learning = models.resnet18(pretrained="ResNet18_Weights.IMAGENET1K_V1").to(DEVICE)

# freeze all params
for params in resnet18_transfer_learning.parameters():
    params.requires_grad_ = False

nr_filters = resnet18_transfer_learning.fc.in_features  # num of input features for fc layer
resnet18_transfer_learning.fc = nn.Linear(nr_filters, 1).to(DEVICE)

# resnet18 = ResNet18(3, 1).to(device)
# vgg16 = VGG(architecture="VGG16", in_channels=3, num_classes=1).to(device)

logistic_reg = LogisticRegression(penalty=None, class_weight="balanced", solver="newton-cg", random_state=0)
svc = SVC(random_state=0)
rf = RandomForestClassifier(max_depth=5, random_state=0)
scaler = MinMaxScaler()


fit_model(imgs, target, scaler, logistic_reg)
fit_model(imgs, target, scaler, svc)
fit_model(imgs, target, scaler, rf)
eval_convnet(model=resnet18_transfer_learning, checkpoint_name="best_model_RESNET18_transfer_learning")
