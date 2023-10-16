import models as model
import augmentations as augment
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import faiss
from sklearn.metrics import roc_curve, auc

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
encoder = model.Encoder().to('cuda')
projection = model.Projector(encoder_dimension).to('cuda')
T = augment.train_transform
T_test = augment.test_transform
loss_Linv, loss_Lcov, loss_Lvar = [], [], []
loss_Linv_test, loss_Lvar_test, loss_Lcov_test = [], [], []
loss_test_inv, loss_test_var, loss_test_cov = [], [], []
parameters = list(encoder.parameters()) + list(projection.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)


# Q1_PART2:
def Q1_part2(encoder):
    PATH = '/content/drive/MyDrive/Colab Notebooks/ex2_aml'
    test_data_MNIST_= torchvision.datasets.MNIST(root=PATH,train=False, download=True, transform=transforms.ToTensor())
    test_data_MNIST = torch.utils.data.DataLoader(test_data_MNIST_, batch_size= batches, shuffle=True)

    test_data_CIFAR10_= torchvision.datasets.CIFAR10(root=PATH,train=False, download=True, transform=transforms.ToTensor())
    test_data_CIFAR10 = torch.utils.data.DataLoader(test_data_CIFAR10_, batch_size= batches, shuffle=True)
    resize = transforms.Resize((32, 32))
    test_images = list()
    all_encoder_MNIST = []
    for batch, label in test_data_MNIST:
        with torch.no_grad():
            resized_images = [resize(image.to('cuda')) for image in batch]
            resized_images = torch.stack(resized_images).repeat(1,3,1,1)
            test_images.extend(resized_images)
            h = encoder.encode(resized_images)
        all_encoder_MNIST.append(h)
    all_encoder_MNIST = torch.cat(all_encoder_MNIST, dim=0)

    all_encoder_CIFAR10 = []
    for batch, label in test_data_CIFAR10:
        with torch.no_grad():
            h = encoder.encode(batch.to('cuda'))
            cifar_image = [image.to('cuda') for image in batch]
            test_images.extend(cifar_image)
        all_encoder_CIFAR10.append(h)
    all_encoder_CIFAR10 = torch.cat(all_encoder_CIFAR10, dim=0)

    represntions = torch.cat((all_encoder_MNIST, all_encoder_CIFAR10), dim=0).cpu().numpy()
    labels = np.concatenate((np.ones(10000), np.zeros(10000)), axis=0)

    ind = faiss.IndexFlatL2(encoder_dimension)
    ind.add(represntions)
    distances, indices = ind.search(represntions, 3)  # k=2

    # Compute kNN density estimation
    densities = 1/ (distances[:, 1:].sum(axis=1)) / 2
    return test_images,densities,labels

# Q2_PART2:
def Q2_part2(labels,densities_arr):

    for i in range(2):
        fpr, tpr, thresholds_method = roc_curve(labels, densities_arr[i])
        auc_method = auc(fpr, tpr)
        models = ['vicReg','NoGeneratedNeighbor']
        plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(models[i], float(auc_method)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# Q3_PART2:
def Q3_part2(encoder1,encoder6):
    test_images1,d1,labels= Q1_part2(encoder1)
    test_images6,d6,labels= Q1_part2(encoder6)

    sorted_indices_1 = np.argsort(d1)[-7:]
    sorted_indices_6 = np.argsort(d6)[-7:]

    fig, axs = plt.subplots(2, 7)
    for im in range(7):
      axs[0, im].imshow(test_images1[sorted_indices_1[im]].cpu().permute(1, 2, 0))
      axs[1, im].imshow(test_images6[sorted_indices_6[im]].cpu().permute(1, 2, 0))

    plt.show()

if __name__ == '__main__':
    encoder1 = model.Encoder()
    encoder1.load_state_dict(torch.load('ex2/encoder_parameters1_20_learning.pth',
                                        map_location=torch.device('cpu')))
    test_images1, d1, labels = Q1_part2(encoder1)
    sorted_indices_1 = np.argsort(d1)[-7:]
    fig, axs = plt.subplots(1, 7)
    for im in range(7):
        axs[0, im].imshow(test_images1[sorted_indices_1[im]].cpu().permute(1, 2, 0))
    plt.show()

