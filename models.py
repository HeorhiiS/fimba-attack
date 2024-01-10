
'''Disclaimer:
Please note that some hyperparameters can differ from the ones used in the paper.
'''

import os
import numpy as np
import pandas as pd
import contextlib
from sklearn.metrics import confusion_matrix, accuracy_score

#utils
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from math import ceil, sqrt
from PIL import Image as im
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import random
import time

#Deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F


@contextlib.contextmanager  # custom change dir context switcher
def change_dir(path):
    _oldcwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(_oldcwd)


def closestDivisors(n):  # finds two biggest fractions
    a = round(sqrt(n))
    while n % a > 0: a -= 1
    return a, n // a


def vec_to_img(vector):  # convert 1d array to 2d representation

    shape = closestDivisors(vector.shape[1])
    reshaped_vec = vector.reshape(shape)
    return reshaped_vec


def display_vec(vector):  # display vectorized representation of numeric data as an image

    norm = Normalize(vmin=min(vector.flatten()), vmax=max(vector.flatten()))
    data_norm = norm(vector)
    plt.imshow(data_norm, cmap='plasma')


class TCGA_Dataset(Dataset):

    def __init__(self, data, targets, transform=True, final_dim=2, scale=True, scale_range=(-1, 1)):
        self.og_index = data.index
        self.data = data.reset_index(drop=True)
        self.target = targets.reset_index(drop=True)
        self.scaler = MinMaxScaler(feature_range=scale_range)
        self.scaled_data = None

        numpied_data = self.data.values
        if scale:
            self.scaler.fit(numpied_data)
            scaled_data = self.scaler.transform(numpied_data)
            numpied_data = scaled_data
            self.scaled_data = pd.DataFrame(scaled_data, columns=self.data.columns)

        if transform == True and final_dim == 2:  # if true then transform to 2d tensors
            # min_of_data = self.data.values.min()
            # epsilon = 1e-6
            # scale = np.abs(min_of_data) + epsilon
            # self.scale = scale

            shape = closestDivisors(self.data.shape[1])
            reshaped_data = np.array([a.reshape(shape) for a in numpied_data])
            self.data = torch.tensor(reshaped_data)
            self.target = torch.tensor(self.target.values)

        elif transform == True and final_dim == 1:
            numpied_data = self.data.values
            self.data = torch.tensor(numpied_data)
            self.target = torch.tensor(self.target.values)

    def __shape__(self):
        return self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __invert_scaler__(self):
        return self.scaler.inverse_transform(self.data.values)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class COVID_Dataset(Dataset): # This is a wrapper class for the COVID dataset

    def __init__(self, data, targets, transform=True, final_dim=2, scale=True, scale_range=(-1, 1)):
        self.og_index = data.index
        self.data = data.reset_index(drop=True)
        self.target = targets.reset_index(drop=True)
        self.scaler = MinMaxScaler(feature_range=scale_range)
        self.scaled_data = None

        numpied_data = self.data.values
        if scale: # if true then scale the data
            self.scaler.fit(numpied_data)
            scaled_data = self.scaler.transform(numpied_data)
            numpied_data = scaled_data
            self.scaled_data = pd.DataFrame(scaled_data, columns=self.data.columns)

        if transform == True and final_dim == 2: # if true then transform to 2d tensors, used for VAE

            shape = closestDivisors(self.data.shape[1])
            reshaped_data = np.array([a.reshape(shape) for a in numpied_data])
            self.data = torch.tensor(reshaped_data)
            self.target = torch.tensor(self.target.values)

        elif transform == True and final_dim == 1: # if true then transform to 1d tensors
            numpied_data = self.data.values
            self.data = torch.tensor(numpied_data)
            self.target = torch.tensor(self.target.values)

    def __shape__(self): # returns the shape of the data
        return self.data.shape

    def __len__(self): # returns the length of the data
        return self.data.shape[0]

    def __getitem__(self, idx): # returns the data at the index
        return self.data[idx], self.target[idx]

    def get_data_min(self): # returns the minimum value of the data
        return self.data.min()


class Loader(DataLoader): # This is a wrapper class for the Pytorch DataLoader class
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)


class VariationalEncoder(nn.Module): # Class that describes the encoder of the VAE
    def __init__(self, input_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, 2100)

        self.linear2 = nn.Linear(2100, 1600)
        self.linear3 = nn.Linear(1600, 1200)
        self.linear4 = nn.Linear(1200, 800)
        self.linear5 = nn.Linear(800, 512)

        self.linear2_1 = nn.Linear(512, latent_dims)
        self.linear3_1 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.tanh(self.linear5(x))

        mu = self.linear2_1(x)
        sigma = torch.exp(self.linear3_1(x))
        z = mu + sigma * self.N.sample(mu.shape) # sampling from the normal distribution
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum() #kl term for loss
        return z


class VariationalDecoder(nn.Module): # VAE decoder class
    def __init__(self, latent_dims, output_dims):
        super(VariationalDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 800)
        self.linear3 = nn.Linear(800, 1200)
        self.linear4 = nn.Linear(1200, 1600)
        self.linear5 = nn.Linear(1600, 2100)

        self.linear2_1 = nn.Linear(2100, output_dims)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.tanh(self.linear5(x))

        x = F.tanh(self.linear2_1(x))
        return x


class VariationalAutoencoder(nn.Module): # Class that describes the whole VAE
    def __init__(self, input_dims, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, latent_dims)
        self.decoder = VariationalDecoder(latent_dims, input_dims)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


def render_encoding(autoencoder, dataloader): # Function that renders the encoding of the VAE
    sample_patches = []
    for patch in iter(dataloader):
        sample_patches = patch
        break


def train_encoder(autoencoder, dataloader, epochs=50): # Function that trains the VAE
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-5)
    criterion = nn.BCELoss()
    print('Starting the training...')
    autoencoder.train()
    loss_during_training = []

    sample_patches = []
    for patch in iter(dataloader):
        sample_patches = patch
        break

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data = data.to(torch.float32)
            optimizer.zero_grad()
            recon_batch = autoencoder(data)

            data = torch.flatten(data, start_dim=1)
            # loss = criterion(recon_batch, data) # binary cross entropy loss, but we are using mse+kld
            loss = ((data - recon_batch) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            train_loss += loss.item()

            # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 2) # gradient clipping just in case
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(tcga_data_train)))

        loss_during_training.append(train_loss / len(tcga_data_train))
    return np.array(loss_during_training)


def loss_function(x, x_hat, mean, log_var): # VAE custom loss function
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def test_encoder(autoencoder, dataloader): # evaluation function for VAE
    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            recon_batch = autoencoder(data)
            # test_loss += F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum').item()
            test_loss += ((data - recon_batch) ** 2).sum().item()
    test_loss /= len(data.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def interpolate_encoder(autoencoder, x1, x2, n=10): # function that performs interpolation between two images
    z1 = autoencoder.encoder(x1)
    z2 = autoencoder.encoder(x2)
    # z = torch.stack([z1 + (z2 - z1)*i/n for i in range(n+1)])
    z = torch.stack([z1 + (z2 - z1) * i for i in np.linspace(0, 1, n + 1)])

    return autoencoder.decoder(z).to('cpu').detach().numpy()

'''
Below are other machine learning classes that were used in the project, and their respective utility functions
'''
class FastCNN(nn.Module):
    def __init__(self, num_classes=1):
        self.afterconv_dim = 1
        super(FastCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(512)
        # self.fc1 = nn.Linear(64 * 10 * self.afterconv_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


def cnn_train(model, train_dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            images = images.unsqueeze(1)
            labels = labels.unsqueeze(1)
            outputs = model(images.to(device).to(torch.float32))

            labels = labels.to(device).to(torch.float32)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, train_loss / len(train_dataloader)))


def cnn_eval(model, test_dataloader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.unsqueeze(1)
            labels = labels.to(device)
            outputs = model(images.to(device).to(torch.float32))
            predictions = (outputs > 0.5).float()
            correct += (torch.squeeze(predictions) == torch.squeeze(labels)).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print("Accuracy on test set: {:.4f}".format(accuracy))


class BasicBlock(nn.Module): # This is the basic block for the ResNet18 model

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EfficientResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(EfficientResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * 1  # block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out


def resnet_train(model, train_dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            images = images.unsqueeze(1)
            labels = labels.unsqueeze(1)
            outputs = model(images.to(device).to(torch.float32))

            labels = labels.to(device).to(torch.float32)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, train_loss / len(train_dataloader)))


def resnet_eval(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.unsqueeze(1)
            labels = labels.to(device)
            outputs = model(images.to(device).to(torch.float32))
            predictions = (outputs > 0.5).float()
            correct += (torch.squeeze(predictions) == torch.squeeze(labels)).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print("Accuracy on test set: {:.4f}".format(accuracy))


class DeepVisionTransformer(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepVisionTransformer, self).__init__()

        # Input size: (batch_size, 1, 43, 67)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Flatten the output from convolutional layers

        self.fc = nn.LazyLinear(512)

        # Transformer block 1
        self.transformer_block1 = nn.Transformer(d_model=512, nhead=8, batch_first=False)

        # Transformer block 2
        # self.transformer_block2 = nn.Transformer(d_model=512, nhead=8, batch_first=False)

        # Output layer
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass input through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output from convolutional layers

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        # Pass input through the first transformer block

        x = x.unsqueeze(0)  # Add a fake sequence length dimension
        tgt = x
        x = self.transformer_block1(x, tgt).squeeze(0)

        # Pass input through the second transformer block
        # x = x.unsqueeze(0) # Add a fake sequence length dimension
        # tgt = x
        # x = self.transformer_block2(x, tgt).squeeze(0)

        # Pass through the output layer
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x

'''
Since we can't use the same inbuilt predict function (like in XGBoost or RF) for the deep learning models,
we have to write our own.
'''
def df_dl_predict(model, df, label_set, ds_type, device):
    if ds_type == 'tcga':
        ds = TCGA_Dataset(df, label_set, scale=False, scale_range=(-1,1))
    else:
        ds = COVID_Dataset(df, label_set, scale=False, scale_range=(-1,1))

    testloader = Loader(ds, batch_size=10, shuffle=False)

    loader = testloader
    preds = []
    model = model
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            images = images.unsqueeze(1)
            labels = labels.to(device)
            outputs = model(images.to(device).to(torch.float32))
            predictions = (outputs > 0.5).float()
            correct += (torch.squeeze(predictions) == torch.squeeze(labels)).sum().item()
            total += labels.size(0)

            preds.append(torch.squeeze(predictions).cpu().detach().numpy())

        accuracy = correct / total
        # print("Accuracy on test set: {:.4f}".format(accuracy))

    preds = np.array(preds)
    return np.hstack(preds)


def df_dl_predict2(model, df, label_set, ds_type, device): # deprecated because of the new df_dl_predict function above
    preds = []
    model = model
    with torch.no_grad():
        correct = 0
        total = 0
        for index, row in df.iterrows():
            row = np.array(row)

            shape = closestDivisors(row.shape[0])
            images = torch.tensor(row.reshape(shape))

            images = images.unsqueeze(0).unsqueeze(0)
            # labels = labels.to(device)
            outputs = model(images.to(device).to(torch.float32))

            predictions = (outputs > 0.5).float()
            # if index == 1162:
            #     print(images.to(device).to(torch.float32))
            #     print(predictions)
            #     est = np.array(xtest.iloc[1161])
            #     print(torch.tensor(test.reshape(shape)).to(device).to(torch.float32))

            # correct += (torch.squeeze(predictions) == torch.squeeze(labels)).sum().item()
            # total += labels.size(0)
            preds.append(torch.squeeze(predictions).cpu().detach().numpy())

        # accuracy = correct / total
        # print("Accuracy on test set: {:.4f}".format(accuracy))
    preds = np.array(preds)
    return np.hstack(preds)

'''
Below are the attack functions for the different models. (*args) is used to pass in the arguments implicitly, because
this function is called in a multiprocessing pool, and the pool can't pass in the arguments explicitly.

Arg explanation:
topf: the top features to be attacked, the number of features changed in the plots
shap_importance: the shapley values for the features, sorted by importance in descending order
model: the model to be attacked
neg_data_test: the negative class test data
pos_data_test: the positive class test data
xtest: the test data
ytest: the test labels
tree_fps: KDTree fit on false positives
tree_fns: KDTree fit on false negatives
fps: false positives, in a form of dataframe
fns: false negatives, in a form of dataframe
increase_fn: the flag to indicate whether to increase false negatives or not
increase_fp: the flag to indicate whether to increase false positives or not
model_type: the type of the model, either 'shallow' or 'deep', to tell the function which predict function to use:
df_dl_predict() for deep learning models, and model.predict() for shallow models

Note that for simpliciy I pass the shapley values as an argument, but we could have also passed in the top features or computed shap values in the function
either outside the main loop or compute them in real time. The resulting complexity does not differ much, but I chose to pass in the shap values as an argument.
runatk_standalone has an inloop conpuation example.
'''

def shap_attack_mp(*args):  #FP/FN Limit attack function
    topf, shap_importance, model, neg_data_test, pos_data_test, xtest, ytest, tree_fps, tree_fns, fps, fns, increase_fn, increase_fp, model_type = args

    FP_count = [] # Make arrays for storing the number of false positives and false negatives
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy() # Sometimes there is a memory leak, so we need to get a fresh set of data by copying a dataframe
    precision = 10 # interpolation precision, how many values are there going to be between the a1 and a2 value of the feature
    increase_fp = increase_fp
    increase_fn = increase_fn
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)


    confm_upd = confusion_matrix(ytest, preds) # Get the confusion matrix, this is our baseline values before the attack
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds)) # Get the initial accuracy score

    #IMPORTANT: topf is the number of features to be attacked
    top_feat = shap_importance.iloc[:topf] # Get the shap values of top features
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index # get indices of the negative class test data
    one_idx_test = pos_data_test.index # get indices of the positive class test data

    print(f'Running SHAP attack on {topf} features...')
    start_time = time.perf_counter() # Start the timer, used to measure runtime of the attack if needed
    for index, row in x_copy.iterrows():  # Iterate over the test data

        if increase_fp: # If we are increasing false positives
            if index in zero_idx_test: # If the index is in the negative class test data,

                id_neg = tree_fps.query(row, k=2) # Get the closest FP vector

                if id_neg[0][
                    0] == 0:  # this means we are getting FP vector, but negative label can also be an FP vector...
                    continue # if the closest FP vector is the same as the original vector, then skip this iteration
                else:
                    vector_id = id_neg[1][0] # Get the index of the closest FP vector
                    a0 = row # original vector
                    a1 = fps.iloc[vector_id] # closest FP vector

                    for feature in top_feat_names: # Iterate over the top features
                        a0_val = a0[feature] # Get the value of the feature in the original vector
                        a1_val = a1[feature] # Get the value of the feature in the closest FP vector
                        sample_space = np.linspace(a0_val, a1_val, precision) # Get the sample space between the two vectors
                        x_copy.at[index, feature] = sample_space[-2] # Change the value of the feature in the original vector to the second last value in the sample space


        if increase_fn: # If we are increasing false negatives
            if index in one_idx_test:  # If the index is in the positive class test data,

                id_neg = tree_fns.query(row, k=2)

                if id_neg[0][
                    0] == 0:  # this means we are getting FN vector
                    continue  # if the closest FN vector is the same as the original vector, then skip this iteration
                else:
                    vector_id = id_neg[1][0]

                    a0 = row # original vector
                    a1 = fns.iloc[vector_id] # closest FN vector

                for feature in top_feat_names:
                    a0_val = a0[feature] # Get the value of the feature in the original vector
                    a1_val = a1[feature] # Get the value of the feature in the closest FN vector
                    sample_space = np.linspace(a0_val, a1_val, precision) # Get the sample space between the two vectors
                    x_copy.at[index, feature] = sample_space[-2]
    stop_time = time.perf_counter()

    print(f'Running SHAP attack on {topf} features completed in {stop_time - start_time:0.4f} seconds')
    if model_type == 'deep': # Get the predictions of the model
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')
    else:
        preds = model.predict(x_copy)

    confm_upd = confusion_matrix(ytest, preds) # Get the confusion matrix after the attack
    FP_count.append(confm_upd[0][1]) # Get the number of false positives and false negatives after the attack
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds)) # Get the accuracy score after the attack
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')
    return FP_count, FN_count, Accuracy, x_copy # Return the results

'''
For this attack we are still using shap values, but note the difference in the way features get modified.
Here we use random sampling from the uniform distribution to modify the features. Limits of 
the uniform distribution are set to the min and max values of the values in the dataset. This is done for ablation and is not in the paper.
'''

def shap2mod_attack_mp(*args): # Shap modification attack

    topf, shap_importance, model, neg_data_test, pos_data_test, xtest,ytest, tree_fps, tree_fns, increase_fp, increase_fn, model_type= args
    dataset_min = -1 # Set the min and max values of the dataset
    dataset_max = 1

    FP_count = []
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy()
    increase_fp = increase_fp
    increase_fn = increase_fn


    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)


    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))

    top_feat = shap_importance.iloc[:topf]
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index
    one_idx_test = pos_data_test.index
    start_time = time.perf_counter()
    for index, row in x_copy.iterrows():  # Iterate over the test data


        if increase_fn: # if we are increasing false negatives

            if index in one_idx_test:  # if the index is in the positive class test data
                id_neg = tree_fns.query(row, k=2) # Get the closest FN vector, but here we just want to filter out FN vectors

                if id_neg[0][0] == 0:
                    continue
                else:

                    for feature in top_feat_names:
                        x_copy.at[index, feature] = random.uniform(dataset_min, dataset_max) # Change the value of the feature in the original vector to a random value from the uniform distribution


        if increase_fp: # if we are increasing false positives

            if index in zero_idx_test:
                id_pos = tree_fps.query(row, k=2)
                if id_pos[0][0] == 0:
                    continue
                else:
                    for feature in top_feat_names:
                        x_copy.at[index, feature] = random.uniform(dataset_min, dataset_max) # Change the value of the feature in the original vector to a random value from the uniform distribution

    stop_time = time.perf_counter()
    print(f'Running SHAP attack on {topf} features completed in {stop_time - start_time:0.4f} seconds')
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')
    else:
        preds = model.predict(x_copy)
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')

    return FP_count, FN_count, Accuracy, x_copy # Return the results

'''
Same as the FP/FN limit attack, but here we are using the KDTREES fitted on positive and negative class vectors
So the limit becomes true positives and true negative aka the closest oppsite class vector. The results for this are located in technical abstract
'''

def shap2_attack_mp(*args): # TN/TP Limit Attack
    topf, model, shap_importance, neg_data_test, pos_data_test, xtest, ytest, tree_pos_test, tree_neg_test, increase_fp, increase_fn, model_type = args

    FP_count = []
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy()
    increase_fp = increase_fp
    increase_fn = increase_fn
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)

    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    precision = 10

    top_feat = shap_importance.iloc[:topf]
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index
    one_idx_test = pos_data_test.index

    start_time = time.perf_counter()
    for index, row in x_copy.iterrows():

        if increase_fp: # if we are increasing false positives
            if index in zero_idx_test: # if the index is in the negative class test data
                id_pos = tree_pos_test.query(row, k=2) # Get the closest positive vector
                if id_pos[0][0] == 0:
                    continue
                else:
                    vector_id = id_pos[1][0]
                    a0 = row
                    a1 = pos_data_test.iloc[vector_id]

                    for feature in top_feat_names:
                        a0_val = a0[feature]
                        a1_val = a1[feature]
                        sample_space = np.linspace(a0_val, a1_val, precision)
                        x_copy.at[index, feature] = sample_space[-2]

        if increase_fn: # if we are increasing false negatives
            if index in one_idx_test: # if the index is in the positive class test data
                id_neg = tree_neg_test.query(row, k=2) # Get the closest negative vector

                if id_neg[0][0] == 0:
                    continue
                else:
                    vector_id = id_neg[1][0]
                    a0 = row
                    a1 = neg_data_test.iloc[vector_id]
                    for feature in top_feat_names:
                        a0_val = a0[feature]
                        a1_val = a1[feature]
                        sample_space = np.linspace(a0_val, a1_val, precision)
                        x_copy.at[index, feature] = sample_space[-2]
    stop_time = time.perf_counter()
    print(f'Running SHAP attack on {topf} features completed in {stop_time - start_time:0.4f} seconds')
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')

    return FP_count, FN_count, Accuracy, x_copy

'''
This is the baseline attack where we are randomly selecting features to attack. I only use this to compare the results of the other attacks
and it is not present in the paper so you can skip this one.
'''
def shap_attack_baseline_random_mp(*args):
    topf, shap_importance, model, neg_data_test, pos_data_test, xtest, ytest, tree_fps, tree_fns, fps, fns, increase_fn, increase_fp, model_type = args

    FP_count = []
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy()
    precision = 10
    increase_fp = increase_fp
    increase_fn = increase_fn
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))


    '''
    Here we are sampling the top features without any consideration for their importance score
    '''

    top_feat = shap_importance.sample(frac=1).iloc[:topf] # randomize the features
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index
    one_idx_test = pos_data_test.index

    x_copy = xtest.copy()  # get a fresh set
    print(f'Running SHAP attack on {topf} features...')

    for index, row in x_copy.iterrows():  # FP

        if increase_fp:
            if index in zero_idx_test:

                id_neg = tree_fps.query(row, k=2)

                if id_neg[0][
                    0] == 0:
                    continue
                else:
                    vector_id = id_neg[1][0]
                    a0 = row
                    a1 = fps.iloc[vector_id]

                    for feature in top_feat_names:
                        a0_val = a0[feature]
                        a1_val = a1[feature]
                        sample_space = np.linspace(a0_val, a1_val, precision)
                        x_copy.at[index, feature] = sample_space[-2]
                        # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FP vector value {a1_val} | Sample space: {sample_space[-2]}")

        if increase_fn:
            if index in one_idx_test:  # positive labels or 1

                id_neg = tree_fns.query(row, k=2)
                # print(id_neg)
                if id_neg[0][
                    0] == 0:  # this means we are getting fps vector TODO: remove fns from the test set before running the algorithm
                    continue
                else:
                    vector_id = id_neg[1][0]
                    # print(id_neg)

                    a0 = row
                    a1 = fns.iloc[vector_id]

                for feature in top_feat_names:
                    a0_val = a0[feature]
                    a1_val = a1[feature]
                    sample_space = np.linspace(a0_val, a1_val, precision)

                    # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FN vector value {a1_val} | Sample space: {sample_space[-2]}")

                    x_copy.at[index, feature] = sample_space[-2]

    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')
    else:
        preds = model.predict(x_copy)
    # preds=df_dl_predict(model, x_copy,ytest,ds_type='tcga')
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')
    return FP_count, FN_count, Accuracy # x_copy

'''
This is a fully random attack where we are randomly selecting features to attack and randomly selecting the values to attack them with. This is our brute force attack in the paper.
'''

def shap2mod_attack_full_rand_mp(*args):
    topf, shap_importance, model, neg_data_test, pos_data_test, xtest,ytest, tree_fps, tree_fns, increase_fp, increase_fn, model_type= args
    dataset_min = -1
    dataset_max = 1

    FP_count = []
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy()
    increase_fp = increase_fp
    increase_fn = increase_fn


    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')

    else:
        preds = model.predict(x_copy)
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    precision = 10

    top_feat = shap_importance.sample(frac=1).iloc[:topf] # randomize the features
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index
    one_idx_test = pos_data_test.index
    start_time = time.perf_counter()
    for index, row in x_copy.iterrows(): # Iterate over the rows of the dataframe

        if increase_fn: # if we want to increase the number of false negatives

            if index in one_idx_test: # positive labels or 1
                id_neg = tree_fns.query(row, k=2) # get the closest false negative vector
                if id_neg[0][0] == 0:
                    continue
                else:
                    for feature in top_feat_names:
                        x_copy.at[index, feature] =random.uniform(dataset_min, dataset_max) # randomly select a value between the min and max of the dataset

        if increase_fp: # if we want to increase the number of false positives

            if index in zero_idx_test:
                id_pos = tree_fps.query(row, k=2)  # get the closest false positive vector
                if id_pos[0][0] == 0:
                    continue
                else:
                    for feature in top_feat_names:
                        x_copy.at[index, feature] = random.uniform(dataset_min, dataset_max) # randomly select a value between the min and max of the dataset
    stop_time = time.perf_counter()
    print(f'Running SHAP attack on {topf} features completed in {stop_time - start_time:0.4f} seconds')
    print()
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cuda')
    else:
        preds = model.predict(x_copy)
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')

    return FP_count, FN_count, Accuracy, x_copy



if __name__ == '__main__':
    pass