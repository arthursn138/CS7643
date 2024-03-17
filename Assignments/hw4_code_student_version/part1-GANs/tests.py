import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gan_pytorch import preprocess_img, deprocess_img, rel_error, count_params, ChunkSampler
from gan_pytorch import sample_noise
from gan_pytorch import Flatten, Unflatten, initialize_weights
from gan_pytorch import bce_loss, discriminator_loss, generator_loss
from gan_pytorch import get_optimizer, eval_gan, eval_discriminator
from gan_pytorch import generator, discriminator

answers = dict(np.load('gan-checks.npz'))
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class test_gan(unittest.TestCase):
  def test_sample_noise(self):
    batch_size = 3
    dim = 4
    torch.manual_seed(6476)
    z = sample_noise(batch_size, dim)
    np_z = z.cpu().numpy()
    assert np_z.shape == (batch_size, dim)
    assert torch.is_tensor(z)
    assert np.all(np_z >= -1.0) and np.all(np_z <= 1.0)
    assert np.any(np_z < 0.0) and np.any(np_z > 0.0)
    print('All tests passed!')


  def test_discriminator_loss(self):
    logits_real, logits_fake, d_loss_true = answers['logits_real'], answers['logits_fake'],answers['d_loss_true']

    d_loss = discriminator_loss(torch.Tensor(logits_real).type(dtype),
                                torch.Tensor(logits_fake).type(dtype)).cpu()
    torch.allclose(torch.Tensor(d_loss_true), torch.Tensor(d_loss), rtol=1e-05, atol=1e-08, equal_nan=False)
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss.numpy()))

  def test_generator_loss(self):
    sample_logits = torch.tensor([0.0, 0.5, 0.4, 0.3])
    g_loss = generator_loss(torch.Tensor(sample_logits).type(dtype)).cpu().numpy()
    torch.allclose(torch.Tensor([0.5586487]), torch.Tensor(g_loss), rtol=1e-05, atol=1e-08, equal_nan=False)
    print("Maximum error in d_loss: %g"%rel_error(0.5586487, g_loss.item()))

  def test_bce_loss(self):
    input = torch.tensor([0.4, 0.1, 0.9, 0.5])
    target = torch.tensor([0.3, 0.25, 0.75, 0.6])
    bc_loss = bce_loss(input, target).item()
    torch.allclose(torch.Tensor([0.6882]), torch.Tensor([bc_loss]), rtol=1e-05, atol=1e-08, equal_nan=False)
    print("Maximum error in d_loss: %g"%rel_error(0.6882, bc_loss))

  def test_discriminator(self):
      model = discriminator()
      cur_count = count_params(model)
      print('Number of parameters in discriminator: ', cur_count)

  def test_generator(self):
      model = generator()
      cur_count = count_params(model)
      print('Number of parameters in generator: ', cur_count)

if __name__ == "__main__":
  unittest.main()




  



