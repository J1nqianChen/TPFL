import copy
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision.models import VisionTransformer

from trainmodel.resnet import ResNet

batch_size = 16


def rearrange_model(model):
    if isinstance(model, FedAvgCNN):
        return _split_fedavgcnn(model)
    elif isinstance(model, HARCNN):
        return _split_harcnn(model)
    elif isinstance(model, torchvision.models.resnet.ResNet):
        return _split_resnet(model)
    elif isinstance(model, ResNet):
        return _split_resnet(model)
    elif isinstance(model, VisionTransformer):
        return _split_vit(model)
    elif isinstance(model, TextCNN):
        return _split_textcnn(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _split_fedavgcnn(model):
    # 创建深度拷贝避免修改原始模型
    model_copy = copy.deepcopy(model)

    # 构建新的head
    head = nn.Sequential(
        copy.deepcopy(model_copy.fc1),
        copy.deepcopy(model_copy.fc)
    )

    # 替换原始层为Identity
    model_copy.fc1 = nn.Identity()
    model_copy.fc = nn.Identity()

    return BaseHeadSplit(model_copy, head)


def _split_harcnn(model):
    model_copy = copy.deepcopy(model)
    head = copy.deepcopy(model_copy.fc)
    model_copy.fc = nn.Identity()
    return BaseHeadSplit(model_copy, head)


def _split_textcnn(model):
    model_copy = copy.deepcopy(model)
    head = copy.deepcopy(model_copy.fc)
    model_copy.fc = nn.Identity()
    return BaseHeadSplit(model_copy, head)

def _split_resnet(model):
    model_copy = copy.deepcopy(model)
    head = copy.deepcopy(model_copy.fc)
    model_copy.fc = nn.Identity()
    return BaseHeadSplit(model_copy, head)


def _split_vit(model):
    model_copy = copy.deepcopy(model)
    head = copy.deepcopy(model_copy.heads)
    model_copy.heads = nn.Identity()
    return BaseHeadSplit(model_copy, head)

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64 * 26, num_classes=6, conv_kernel_size=(1, 9),
                 pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # logging.info(x.shape)
        out = self.conv1(x)
        # logging.info(out.shape)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


class BaseHeadNormSplit(nn.Module):
    def __init__(self, base, head, norm):
        super(BaseHeadNormSplit, self).__init__()

        self.base = base
        self.head = head
        self.norm = norm

    def forward(self, x):
        out = self.base(x)
        out = self.norm(out)
        out = self.head(out)

        return out


class LocalModel(nn.Module):
    def __init__(self, base, predictor):
        super(LocalModel, self).__init__()

        self.base = base
        self.predictor = predictor

    def forward(self, x):
        out = self.base(x)
        out = self.predictor(out)

        return out


# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out1 = self.fc1(out)
        out2 = self.fc(out1)
        if feature:
            return out2, out1
        return out2


# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

class DNN_dropout(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN_dropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LeNet(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_classes=10, iswn=None, in_channels=1):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out


# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.8,
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out


# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input

#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output


class DropCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.drop1(self.fc1(out))
        out = self.drop2(self.fc2(out))
        out = self.fc(out)
        return out


class ModifiedResNet50(torchvision.models.ResNet):
    def __init__(self, p=0.5, num_classes=1000):
        super(ModifiedResNet50, self).__init__(block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3],
                                               num_classes=num_classes)
        self.dropout = torch.nn.Dropout(p=p)
        self.fc = torch.nn.Linear(2048, num_classes)  # Modify the last fc layer

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Generator_ACGan(nn.Module):

    def __init__(self, dataset, latent_dim, noise_label_combine, num_classes):
        super(Generator_ACGan, self).__init__()

        self.data_set = dataset
        self.latent_dim = latent_dim
        self.noise_label_combine = noise_label_combine
        self.n_classes = num_classes

        if self.noise_label_combine in ['cat']:
            input_dim = 2 * self.latent_dim
        elif self.noise_label_combine in ['cat_naive']:
            input_dim = self.latent_dim + self.n_classes
        else:
            input_dim = self.latent_dim

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))

        self.layer4_1 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                                      nn.Tanh())

        self.layer4_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                    nn.Tanh())

        self.layer4_3 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
                                      nn.Tanh())

        self.layer4_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(True))

        self.layer4_5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.layer4_6 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                      nn.Tanh())
        self.embedding = nn.Embedding(self.n_classes, self.latent_dim)

    def forward(self, noise, label):

        if self.noise_label_combine == 'mul':
            label_embedding = self.embedding(label)
            h = torch.mul(noise, label_embedding)
        elif self.noise_label_combine == 'add':
            label_embedding = self.embedding(label)
            h = torch.add(noise, label_embedding)
        elif self.noise_label_combine == 'cat':
            label_embedding = self.embedding(label)
            h = torch.cat((noise, label_embedding), dim=1)
        elif self.noise_label_combine == 'cat_naive':
            label_embedding = Variable(torch.cuda.FloatTensor(len(label), self.n_classes))
            label_embedding.zero_()
            label_embedding.scatter_(1, label.view(-1, 1), 1)
            h = torch.cat((noise, label_embedding), dim=1)
        else:
            label_embedding = noise
            h = noise

        x = h.view(-1, h.shape[1], 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.data_set in ['Tiny-Imagenet', 'FOOD101']:
            x = self.layer4_2(x)
            x = self.layer5(x)
        elif self.data_set in ['FMNIST', 'MNIST']:
            x = self.layer4_3(x)
        elif self.data_set in ['']:
            x = self.layer4_4(x)
            x = self.layer4_5(x)
            x = self.layer4_6(x)
        else:
            x = self.layer4_1(x)

        return x, h, label_embedding


class DualEncoder(nn.Module):
    def __init__(self, base, head, data_str='Cifar10', model_str='cnn'):
        super().__init__()
        self.g_base = copy.deepcopy(base)
        self.p_base = copy.deepcopy(base)
        # self.posterior = nn.Sequential(nn.Linear(1024, 512),
        #                                nn.ReLU(),
        #                                nn.Linear(512, 512)).cuda()

        logging.info(data_str)
        if 'Cifar' in data_str or 'cifar' in data_str:
            # decode size: 32x32
            self.decoder = Decoder(out_channels=3, dim=1024).cuda()
        elif 'TinyImageNet' in data_str:
            logging.info('tiny')
            self.decoder = Decoder(out_channels=3, dim=4096).cuda()
            if model_str == 'resnet50':
                self.decoder = Decoder(out_channels=3, dim=4096, in_features=2048).cuda()
        elif 'fmnist' in data_str:
            logging.info('fmnist')
            self.decoder = Decoder(out_channels=1, dim=3136).cuda()
        elif 'har' in data_str:
            logging.info('har')
            self.decoder = Decoder(out_channels=9, in_features=1664, dim=1024).cuda()
        elif 'pamap' in data_str:
            logging.info('pamap')
            self.decoder = Decoder(out_channels=9, in_features=3712, dim=6144).cuda()
        else:
            raise NotImplementedError
        self.head = copy.deepcopy(head)

    def forward(self, x, feature_out=False, recon_out=False):
        g_feature = self.g_base(x)
        # logging.info(g_feature.shape)
        bias = self.p_base(x)
        # bias = self.posterior(torch.cat([p_feature, g_feature], dim=1))
        fused_feature = (g_feature + bias)
        output = self.head(fused_feature)
        if feature_out:
            if recon_out:
                reconstruct_image = self.decoder(fused_feature)
                return output, g_feature, bias, reconstruct_image
            return output, g_feature, bias
        else:
            return output


# class DualEncoder(nn.Module):
#     def __init__(self, base, head):
#         super().__init__()
#         self.g_base = copy.deepcopy(base)
#         self.p_base = copy.deepcopy(base)
#         self.posterior = nn.Sequential(nn.Linear(1024, 512),
#                                        nn.ReLU(),
#                                        nn.Linear(512, 512)).cuda()
#
#         self.q_mlp = nn.Sequential(
#             nn.Linear(in_features=512, out_features=128),
#             nn.LayerNorm(128),  # Normalization after Linear layer
#             nn.ReLU()
#         ).cuda()
#
#         self.k_mlp = nn.Sequential(
#             nn.Linear(in_features=512, out_features=128),
#             nn.LayerNorm(128),  # Normalization after Linear layer
#             nn.ReLU()
#         ).cuda()
#
#         self.v_mlp = nn.Sequential(
#             nn.Linear(in_features=512, out_features=128),
#             nn.LayerNorm(128),  # Normalization after Linear layer
#             nn.ReLU()
#         ).cuda()
#         self.head = copy.deepcopy(head)
#         self.attention_fusion = FeatureFusionAttention(dim=512).cuda()
#         self.decoder = Decoder(out_channels=3).cuda()
#
#     def forward(self, x, feature_out=False, reconstruction_out=False):
#         g_feature = self.g_base(x)
#         p_feature = self.p_base(x)
#         bias = self.posterior(torch.cat([g_feature, p_feature], dim=1))
#
#         Q = self.q_mlp(g_feature)
#         K = self.k_mlp(bias)
#         V = self.v_mlp(bias)
#         fused_feature = self.attention_fusion(Q, K, V)
#
#
#         output = self.head(fused_feature)
#         if feature_out:
#             if reconstruction_out:
#                 reconstruction_x = self.decoder(fused_feature)
#                 return output, g_feature, bias, reconstruction_x
#             return output, g_feature, bias
#
#         else:
#             return output


class FeatureFusionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 1.0 / (dim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        # self.layer_norm = nn.LayerNorm(dim)
        self.out_mlp = nn.Sequential(nn.Linear(128, 512),
                                     nn.Dropout(p=0.1))

    def forward(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = self.softmax(attn_scores)
        # attn_weights = self.layer_norm(attn_weights)  # Apply normalization to attention weights
        output = torch.matmul(attn_weights, V)
        output = self.out_mlp(output)
        return output


class Decoder(nn.Module):
    def __init__(self, in_features=512, dim=1024, out_channels=3):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        # Fully connected layer to reverse the flatten operation
        self.fc = nn.Sequential(
            nn.Linear(in_features, dim),
            nn.ReLU(inplace=True)
        )

        # Transposed convolution layers with BatchNorm
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        if out_channels == 9 and dim == 1024:
            logging.info('har')
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=(3, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        elif out_channels == 9 and dim == 6144:
            logging.info('pamap')
            logging.info('pamap')
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        if out_channels == 9 and dim == 1024:
            logging.info('har')
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        if out_channels == 9 and dim == 6144:
            logging.info('pamap')
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=(1, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(out_channels),  # Depending on your normalization needs
            nn.Tanh()  # Use sigmoid for normalized outputs (if input is normalized between 0 and 1)
        )

        if out_channels == 1:
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(out_channels),  # Depending on your normalization needs
                nn.Tanh()  # Use sigmoid for normalized outputs (if input is normalized between 0 and 1)
            )
            logging.info('fmnist')
        elif out_channels == 9 and dim==1024:
            logging.info('har')
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(16, out_channels, kernel_size=(3, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(out_channels),  # Depending on your normalization needs
                nn.Tanh()  # Use sigmoid for normalized outputs (if input is normalized between 0 and 1)
            )
        elif out_channels == 9 and dim==6144:
            logging.info('har')
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(16, out_channels, kernel_size=(1, 4), stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(out_channels),  # Depending on your normalization needs
                nn.Tanh()  # Use sigmoid for normalized outputs (if input is normalized between 0 and 1)
            )

    def forward(self, x):
        # Reverse the fully connected layers
        x = self.fc(x)
        if x.shape[1] == 1024:
            # Reshape to match the input dimensions before flattening in the encoder
            if self.out_channels == 3:
                x = x.view(x.size(0), 64, 4, 4)  # Reshape back to [B, 64, 4, 4]  Cifar
            elif self.out_channels == 9:
                x = x.view(x.size(0), 64, 1, 16)
        elif x.shape[1] == 4096:
            x = x.view(x.size(0), 64, 8, 8)  # TinyImageNet
        elif x.shape[1] == 3136:
            x = x.view(x.size(0), 64, 7, 7)  # FashionMNIST
        elif x.shape[1] == 6144:
            x = x.view(x.size(0), 64, 3, 32)
        else:
            raise NotImplementedError
        # Apply transposed convolutions with BatchNorm
        x = self.deconv1(x)  # Output size: [B, 32, 8, 8]
        x = self.deconv2(x)  # Output size: [B, 16, 16, 16]
        x = self.deconv3(x)  # Output size: [B, 3, 32, 32]
        # logging.info('Deconv')
        # logging.info(x.shape)

        return x



if __name__ == '__main__':
    decoder = Decoder(in_features=512, dim=4096)
    x = torch.randn(128, 512)
    image = decoder(x)
    # logging.info(image.shape)
