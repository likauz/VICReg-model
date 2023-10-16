import models
import models as model
import augmentations as augment
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import faiss



l = 1
eps = 1e-4
lamda = 25
mu = 25
v = 1
projection_dimension_d = 512
encoder_dimension = 128
batches = 256
learning_rate = 7e-3
weight_decay = 1e-6
epochs = 20  # 10 < epochs < 30
# encoder = model.Encoder().to('cuda')
# projection = model.Projector(encoder_dimension).to('cuda')
T = augment.train_transform
T_test = augment.test_transform
loss_Linv, loss_Lcov, loss_Lvar = [], [], []
loss_Linv_test, loss_Lvar_test, loss_Lcov_test = [], [], []
loss_test_inv, loss_test_var, loss_test_cov = [], [], []


PATH = '/Users/leeuziel/Desktop/aml/ex2'
train_data_ = torchvision.datasets.CIFAR10(root=PATH,train=True, download=True, transform=transforms.ToTensor())
train_data = torch.utils.data.DataLoader(train_data_, batch_size= batches, shuffle=True, num_workers=2)
test_data = torchvision.datasets.CIFAR10(root=PATH,train=False, download=True, transform=transforms.ToTensor())
test_data = torch.utils.data.DataLoader(test_data, batch_size= batches, shuffle=False)

def Lvar(z):
    sigma = torch.sqrt(torch.var(z, dim=0) + eps)
    Lv = torch.sum(torch.clamp(v - sigma, min=0)) / projection_dimension_d
    return Lv

def Lcov(z):
    cov_z = torch.cov(z.T).fill_diagonal_(0)
    Lc = torch.sum(torch.pow(cov_z, 2)) / projection_dimension_d
    return Lc

def loss_VICReg(z1,z2,train=False):
    Li = func.mse_loss(z1, z2)
    Lv = (Lvar(z1)+ Lvar(z2))
    Lc = (Lcov(z1)+ Lcov(z2))
    if train:
      loss_Linv.append(Li.tolist())
      loss_Lvar.append(Lv.tolist())
      loss_Lcov.append(Lc.tolist())
    else:
      loss_Linv_test.append(Li.tolist())
      loss_Lvar_test.append(Lv.tolist())
      loss_Lcov_test.append(Lc.tolist())
    loss = lamda * Li + mu * Lv + v * Lc
    return loss


# Train the VICReg model
def train_VICReg(encoder, projection, optimizer):
  # epoch loop
  for ep in tqdm(range(epochs)):
      # batch loop
      for (batch, labels) in train_data:
          optimizer.zero_grad()
          image_batch_T1 = torch.stack([T(image) for image in batch])
          image_batch_T2 = torch.stack([T(image) for image in batch])
          Z1 = projection(encoder.encode(image_batch_T1))
          Z2 = projection(encoder.encode(image_batch_T2))
          loss = loss_VICReg(Z1, Z2, train=True)
          loss.backward()
          optimizer.step()
# test loss
      for batch, labels in test_data:
          optimizer.zero_grad()
          image_batch_T1 = torch.stack([T(image) for image in batch])
          image_batch_T2 = torch.stack([T(image) for image in batch])
          h1 = projection(encoder.encode(image_batch_T1))
          h2 = projection(encoder.encode(image_batch_T2))
          loss_VICReg(h1, h2, train=False)
      loss_test_inv.append(sum(loss_Linv_test)/len(loss_Linv_test))
      loss_test_var.append(sum(loss_Lvar_test)/len(loss_Lvar_test))
      loss_test_cov.append(sum(loss_Lcov_test)/len(loss_Lcov_test))
      loss_Linv_test.clear()
      loss_Lvar_test.clear()
      loss_Lcov_test.clear()

# Q1
def Q1():
    training_batches_long = len(loss_Linv)
    training_batches = torch.linspace(start=0, end=training_batches_long, steps=training_batches_long,dtype=torch.int).tolist()
    test_epochs = torch.linspace(start=0, end=training_batches_long, steps=epochs,dtype=torch.int).tolist()

    inv_train = go.Figure()
    inv_train.add_trace(go.Scatter(x=training_batches, y=loss_Linv, mode='lines', name='Loss Inv'))
    inv_train.add_trace(go.Scatter(x=test_epochs, y=loss_test_inv, mode='lines', name='Test Loss Inv'))
    inv_train.update_layout(title='Invariance loss as a function of batches', xaxis_title='Batches', yaxis_title= 'Invariance Loss')
    inv_train.show()

    var_train = go.Figure()
    var_train.add_trace(go.Scatter(x=training_batches, y=loss_Lvar, mode='lines', name='Loss var'))
    var_train.add_trace(go.Scatter(x=test_epochs, y=loss_test_var, mode='lines', name='Test Loss var'))
    var_train.update_layout(title='Variance loss as a function of batches', xaxis_title='Batches', yaxis_title= 'Variance Loss')
    var_train.show()

    cov_train = go.Figure()
    cov_train.add_trace(go.Scatter(x=training_batches, y=loss_Lcov, mode='lines', name='Loss cov'))
    cov_train.add_trace(go.Scatter(x=test_epochs, y=loss_test_cov, mode='lines', name='Test Loss cov'))
    cov_train.update_layout(title='Covariance loss as a function of batches', xaxis_title='Batches', yaxis_title= 'Covariance Loss')
    cov_train.show()

    plt.show()


# Q2
def plot_points_with_labels(points, labels, pca=False, tsne=False):
    labels_types = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_types)))

    for i, label in enumerate(labels_types):
        indexs = labels == label
        data_points = np.array(points)[indexs]
        plt.scatter(data_points[:, 0], data_points[:, 1], color=colors[i], label=label, s=1) # 3?

    plt.xlabel('X')
    plt.ylabel('Y')
    if tsne:
      plt.title('TSNE 2D representations, colored by their classes')
    if pca:
      plt.title('PCA 2D representations, colored by their classes')
    plt.legend()
    plt.show()


def Q2(encoder):
    encoder.eval()
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    all_encoder_images = []
    labels_images = []
    with torch.no_grad():
      for batch, labels in test_data:
          image_batch_T = torch.stack([T_test(image) for image in batch])
          encode_images = encoder.encode(image_batch_T)
          all_encoder_images.append(encode_images)
          labels_images.append(labels)
      all_encoder_images = torch.cat(all_encoder_images,dim=0).cpu().numpy()
      labels_images = torch.cat(labels_images,dim=0).cpu().numpy()
      test_pca = pca.fit_transform(all_encoder_images)
      test_tsne = tsne.fit_transform(all_encoder_images)
      plot_points_with_labels(test_pca, labels_images,pca=True)
      plot_points_with_labels(test_tsne, labels_images,tsne=True)


# Q3
def Q3(encoder):
    # Freeze the encoder
    for param in encoder.parameters():
        param.requires_grad = False
    classifier = nn.Linear(128, 10)  # 10 is the number of CIFAR10 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    all_labels = 0
    # train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_data:
            images = images
            labels = labels
            optimizer.zero_grad()
            features = encoder.encode(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    with torch.no_grad():
        sum_accuracy = 0

        for batch, labels in test_data:
            labels = labels
            image_batch_T = torch.stack([T_test(image) for image in batch])
            Z = encoder.encode(image_batch_T)
            predict = classifier(Z)
            predict_val, predict_index = torch.max(predict.data, 1)
            matches = (predict_index == labels)
            sum_accuracy += matches.sum().item()
            all_labels += len(labels)
    print('The accuracy is', (sum_accuracy * 100) / all_labels)


# Q6
def Q6_(encoder6, projection6, encoder, optimizer):
    all_encoder_images = []
    labels = []
    PATH = '/content/drive/MyDrive/Colab Notebooks/ex2_aml'
    train_data_ = torchvision.datasets.CIFAR10(root=PATH,train=True, download=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(train_data_, batch_size= batches, shuffle=False, num_workers=2)

    encoder.eval()
    # one epoch for encoding the images
    for batch, label in train_data:
        with torch.no_grad():
            h = encoder.encode(batch)
        all_encoder_images.append(h)
        labels.append(label)
    all_encoder_images = torch.cat(all_encoder_images, dim=0).cpu().numpy()

    ind = faiss.IndexFlatL2(encoder_dimension)
    ind.add(all_encoder_images)

    # train the model
    # epoch loop
    for (batch, labels) in train_data:
        optimizer.zero_grad()
        batch_Neighbors = []
        image_batch_T = []
        for image in batch:
            with torch.no_grad():
              image_encode = encoder.encode(image.view(1,3,32,32)).cpu().numpy()
              _ , neighbors = ind.search(image_encode, 4) # number of neighbors
              select_neighbor_index = np.random.choice(neighbors[0,1:])
              neighbor, _ = train_data_[select_neighbor_index]
              batch_Neighbors.append(neighbor)
              image_batch_T.append(image)
        image_batch_T = torch.stack(image_batch_T)
        neighbors_batch_T = torch.stack(batch_Neighbors)
        Z = projection6(encoder6.encode(image_batch_T))
        Z_Neighbors = projection6(encoder6.encode(neighbors_batch_T))
        loss = loss_VICReg(Z,Z_Neighbors)
        loss.backward()
        optimizer.step()


# Q8
def Q8(encoder):
    ten_images = []
    indexs = []
    all_encoder_images = []
    all_images_neighbor, all_images_distance = [], []
    PATH = '/Users/leeuziel/Desktop/aml/ex2'
    train_data_ = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(train_data_, batch_size=batches, shuffle=False, num_workers=2)

    labels = train_data_.classes
    for label in labels:
        cur_label = labels.index(label)
        label_indexs = []
        for ind in range(len(train_data_)):
            if train_data_.targets[ind] == cur_label:
                label_indexs.append(ind)
        select_index = np.random.choice(label_indexs)
        indexs.append(select_index)
        image, label = train_data_[select_index]
        ten_images.append((image, label))

    # one epoch for encoding the images
    for batch, label in train_data:
        with torch.no_grad():
            h = encoder.encode(batch)
        all_encoder_images.append(h)
        labels.append(label)
    all_encoder_images = torch.cat(all_encoder_images, dim=0).cpu().numpy()
    ten_encoder = [all_encoder_images[i, :] for i in indexs]
    ten_encoder = torch.tensor(ten_encoder)
    ten_encoder = [image.unsqueeze(0) for image in ten_encoder]

    ind = faiss.IndexFlatL2(encoder_dimension)
    ind.add(all_encoder_images)
    encoder.eval()
    for i in range(len(ten_images)):
        with torch.no_grad():
            _, neighbors = ind.search((torch.tensor(ten_encoder[i]).cpu().numpy()), k=6)
        indexs_neighbor = neighbors[0, 1:6]
        images_neighbor = [train_data_[ind][0] for ind in indexs_neighbor]
        with torch.no_grad():
            _, neighbors = ind.search(torch.tensor(ten_encoder[i]).cpu().numpy(), k=50000)
        all_images_neighbor.append(images_neighbor)
        indexs_distance = neighbors[0, -5:]
        images_distance = [train_data_[ind][0] for ind in indexs_distance]
        all_images_distance.append(images_distance)
    return all_images_neighbor, all_images_distance, ten_images


def plot_images(images, ten_images, s):
    import os
    fig, axs = plt.subplots(10, 6, figsize=(12, 20))
    for i in range(10):
        image, _ = ten_images[i]
        image = image.numpy().transpose((1, 2, 0))
        axs[i, 0].imshow(image)
        axs[i, 0].set_title('Choosen Image')
        for j in range(5):
            neighbor_image = images[i][j].numpy().transpose((1, 2, 0))
            axs[i, j + 1].imshow(neighbor_image)
            axs[i, j + 1].set_title(f'Neighbor - {j + 1}')
        for j in range(6):
            axs[i, j].axis('off')

    fig.tight_layout()

    save_path = os.path.join(PATH, f'plot_{s}.png')
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # # train the vecReg model

    # projection1 = model.Projector(encoder_dimension)
    # parameters = list(encoder1.parameters()) + list(projection1.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    # train_VICReg(encoder1, projection1, optimizer)
    #
    # # train the no neighbor vecReg model
    # encoder6 = model.Encoder().to('cuda')
    # projection6 = model.Projector(encoder_dimension).to('cuda')
    # parameters = list(encoder6.parameters()) + list(projection6.parameters())
    # optimizer6 = torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    # Q6_(encoder6, projection6, encoder1, optimizer6)
    #
    # # plot Q8 images
    # all_images_neighbor1, all_images_distance1, ten_images = Q8(encoder1)
    # plot_images(all_images_neighbor1, ten_images, 'img_n_1')
    # plot_images(all_images_distance1, ten_images, 'img_d_1')
    # all_images_neighbor6, all_images_distance6, ten_images6 = Q8(encoder6)
    # plot_images(all_images_neighbor6, ten_images6, 'img_n_6')
    # plot_images(all_images_distance6, ten_images6, 'img_d_6')

    encoder1 = model.Encoder()
    encoder1.load_state_dict(torch.load('ex2/encoder_parameters1_20_learning.pth', torch.device('cpu')))
    Q3(encoder1)
    all_images_neighbor1, all_images_distance1, ten_images = Q8(encoder1)
    plot_images(all_images_neighbor1, ten_images, 'img')





