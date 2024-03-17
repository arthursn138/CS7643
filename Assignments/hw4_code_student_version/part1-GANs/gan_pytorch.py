import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler

from torch.nn import init
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.models import inception_v3
from torchvision import transforms

import PIL

NOISE_DIM = 96

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def sample_noise(batch_size, dim, seed=None):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size (int): Integer giving the batch size of noise to generate.
    - dim (int): Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    ##############################################################################
    # TODO: Implement Noise                                                      #
    #                                                                            #
    # HINT: torch.rand?                                                          #
    ##############################################################################
    return
    ##############################################################################


def discriminator(seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.

    Inputs:
    - seed
    """

    if seed is not None:
        torch.manual_seed(seed)

    model = None

    ##############################################################################
    # TODO: Implement architecture - Just initialize the model, no forward pass  #
    #       Choice of architecture is up to you - many answers could work.       #
    #                                                                            #
    # HINT: nn.Sequential might be helpful. Flatten() might be helpful.          #
    ##############################################################################

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def generator(noise_dim=NOISE_DIM, seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.

    Inputs:
    - noise_dim (int): This is the dimension of the sampled Noise
    """

    if seed is not None:
        torch.manual_seed(seed)

    model = None

    ##############################################################################
    # TODO: Implement architecture                                               #
    #       Choice of architecture is up to you - many answers could work.       #
    #                                                                            #
    # HINT: nn.Sequential might be helpful, nn.UnFlatten() may be useful         #
    ##############################################################################

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    ##############################################################################
    # TODO: Implement BCELoss                                                    #
    ##############################################################################
    return
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement Disc. Loss                                                 #
    ##############################################################################

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement Generator Loss                                             #
    ##############################################################################

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Optimizer                                                            #
    ##############################################################################
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return optimizer


def train_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=250,
              batch_size=128, noise_size=96, num_epochs=20):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    images = []
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2 * (real_data - 0.5)).type(dtype)

            fake_images = None
            d_total_error = None
            g_error = None
            ##############################################################################
            # TODO: Train GAN, some steps already provided, complete the loop            #
            # 1. Call Noise                                                              #
            # 2. Generate a fake image and see if the discriminator can differentiate.   #
            # 3. Calculate loss, then update the appropriate parameters                  #
            # 4. See if the Generator can fool the discriminator                         #
            # 5. Calculate loss, then update the appropriate parameters                  #
            #                                                                            #
            # HINT: Use the variables above that are set to None                         #
            # HINT: Use D_solver and G_solver                                            #
            ##############################################################################

            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                images.append(imgs_numpy[0:16])

                # numIter = 0
                shw_imgs = imgs_numpy[0:12]
                # img = (img - img.min()) / (img.max() - img.min())
                # print(img)
                shw_imgs = (shw_imgs)/2 + 0.5
                show_images(shw_imgs.reshape(-1, 3, 32, 32))
                plt.show()

                shw_imgs = real_data[0:10]
                # img = (img - img.min()) / (img.max() - img.min())
                # print(img)
                shw_imgs = (shw_imgs)/2 + 0.5
                show_images(shw_imgs.reshape(-1, 3, 32, 32))
                plt.show()
                # numIter += 250

            iter_count += 1

    return images


def eval_discriminator(discriminator, oracle_generator, val_loader, device, noise_size=96, batch_size=128):
    # Initialize the discriminator
    # criterion = nn.BCELoss()

    # Initialize metrics
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    # Test the discriminator
    discriminator.eval()
    for real_images, _ in val_loader:
        real_data = real_images.type(dtype)
        logits_real = discriminator(2 * (real_data - 0.5)).type(dtype)

        real_labels = torch.ones(real_images.size(0), 1).to(device)

        g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
        fake_images = oracle_generator(g_fake_seed)
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)

        # Pass real and fake images through the discriminator
        fake_outputs = discriminator(fake_images).type(dtype)

        # Compute metrics
        true_positives = (logits_real.detach().cpu().numpy()
                          >= 0.5).sum().item()
        true_negatives = (fake_outputs.detach().cpu().numpy()
                          < 0.5).sum().item()
        false_positives = (
            fake_outputs.detach().cpu().numpy() >= 0.5).sum().item()
        false_negatives = (
            logits_real.detach().cpu().numpy() < 0.5).sum().item()

        accuracy += (true_positives + true_negatives) / \
            (real_images.size(0) + fake_images.size(0))
        precision += true_positives / (true_positives + false_positives + 1)
        recall += true_positives / (true_positives + false_negatives + 1)

    # Calculate average metrics
    num_batches = len(val_loader)
    accuracy /= num_batches
    precision /= num_batches
    recall /= num_batches

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return accuracy, precision, recall


def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # If the image is in the shape (3, 32, 32), use transpose to change it to (32, 32, 3).
        if img.shape[0] == 3:
            # print(type(img))
            if type(img) == torch.Tensor:
                img = img.detach().cpu().numpy()
            img = np.transpose(np.array(img), (1, 2, 0))
        plt.imshow(img)
        # plt.show()

    return


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params(model):
    """Count the number of parameters in the model. """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count


###### Extra FID Code if interested to view ########
def calculate_inception_features(data, inception_model, batch_size, device):
    """
    Calculate the features based on Image Data using Inception v3
    """
    features = []
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        # Mean-center and scale to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    inception_model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = transform(batch).to(device)
            logits = inception_model(batch)
            features.append(logits)

    features = torch.cat(features, 0)
    return features


def calculate_fid(features_real, features_fake):
    """
    Based on the inceptionv3 features, apply the formula to calculate FID Score
    More info can be found here:
    https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
    """
    mu1 = torch.mean(features_real, dim=0)
    mu2 = torch.mean(features_fake, dim=0)

    # Calculate the covariance matrix for real and generated features
    cov1 = torch.matmul((features_real - mu1).t(),
                        (features_real - mu1)) / (len(features_real) - 1)
    cov2 = torch.matmul((features_fake - mu2).t(),
                        (features_fake - mu2)) / (len(features_fake) - 1)

    diff = mu1 - mu2

    # Calculate the covariance matrix square root
    sqrt_product = torch.mm(torch.sqrt(cov1), torch.sqrt(cov2))

    if torch.isnan(sqrt_product).any():
        sqrt_product = torch.zeros_like(sqrt_product)

    fid = torch.norm(diff) + torch.trace(cov1 + cov2 - 2.0 * sqrt_product)
    return fid.item()


def eval_gan(D, G, discriminator_loss, generator_loss, loader_val,
             batch_size=128, noise_size=96, images=1000):
    """
    Evaluation Gan on FID Scores
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - loader_val: PyTorch DataLoder
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    """
    images = []
    iter_count = 0
    D.eval()
    G.eval()
    fid = []

    for x, _ in loader_val:
        real_data = x.type(dtype)
        logits_real = D(2 * (real_data - 0.5)).type(dtype)

        g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
        fake_images = G(g_fake_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inception_model = inception_v3(
            pretrained=True, transform_input=False).to(device)

        # Calculate Inception features for real and generated images
        real_features = calculate_inception_features(
            x.numpy(), inception_model, batch_size, device)
        generated_features = calculate_inception_features(fake_images.view(
            batch_size, 3, 32, 32).cpu().detach().numpy(), inception_model, batch_size, device)

        # Calculate FID score
        fid.append(calculate_fid(real_features, generated_features))

    return np.mean(fid)
