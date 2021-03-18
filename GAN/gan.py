# #!/usr/bin/env python
#
# # Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)
# # See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
#
# matplotlib_is_available = True
# try:
#     from matplotlib import pyplot as plt
# except ImportError:
#     print("Will skip plotting; matplotlib is not available.")
#     matplotlib_is_available = False
#
# # Data params
# data_mean = 4
# data_stddev = 1.25
#
# # ### Uncomment only one of these to define what data is actually sent to the Discriminator
# # (name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# # (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
# # (name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
# (name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)
#
# print("Using data [%s]" % (name))
#
#
# # ##### DATA: Target data and generator input data
#
# def get_distribution_sampler(mu, sigma):
#     return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian
#
#
# def get_generator_input_sampler():
#     return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
#
#
# # ##### MODELS: Generator model and discriminator model
#
# class Generator(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, f):
#         super(Generator, self).__init__()
#         self.map1 = nn.Linear(input_size, hidden_size)
#         self.map2 = nn.Linear(hidden_size, hidden_size)
#         self.map3 = nn.Linear(hidden_size, output_size)
#         self.f = f
#
#     def forward(self, x):
#         x = self.map1(x)
#         x = self.f(x)
#         x = self.map2(x)
#         x = self.f(x)
#         x = self.map3(x)
#         return x
#
#
# class Discriminator(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, f):
#         super(Discriminator, self).__init__()
#         self.map1 = nn.Linear(input_size, hidden_size)
#         self.map2 = nn.Linear(hidden_size, hidden_size)
#         self.map3 = nn.Linear(hidden_size, output_size)
#         self.f = f
#
#     def forward(self, x):
#         x = self.f(self.map1(x))
#         x = self.f(self.map2(x))
#         return self.f(self.map3(x))
#
#
# def extract(v):
#     return v.data.storage().tolist()
#
#
# def stats(d):
#     return [np.mean(d), np.std(d)]
#
#
# def get_moments(d):
#     # Return the first 4 moments of the data provided
#     mean = torch.mean(d)
#     diffs = d - mean
#     var = torch.mean(torch.pow(diffs, 2.0))
#     std = torch.pow(var, 0.5)
#     zscores = diffs / std
#     skews = torch.mean(torch.pow(zscores, 3.0))
#     kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
#     final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skews.reshape(1, ), kurtoses.reshape(1, )))
#     return final
#
#
# def decorate_with_diffs(data, exponent, remove_raw_data=False):
#     mean = torch.mean(data.data, 1, keepdim=True)
#     mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
#     diffs = torch.pow(data - Variable(mean_broadcast), exponent)
#     if remove_raw_data:
#         return torch.cat([diffs], 1)
#     else:
#         return torch.cat([data, diffs], 1)
#
#
# def train():
#     # Model parameters
#     g_input_size = 1  # Random noise dimension coming into generator, per output vector
#     g_hidden_size = 5  # Generator complexity
#     g_output_size = 1  # Size of generated output vector
#     d_input_size = 500  # Minibatch size - cardinality of distributions
#     d_hidden_size = 10  # Discriminator complexity
#     d_output_size = 1  # Single dimension for 'real' vs. 'fake' classification
#     minibatch_size = d_input_size
#
#     d_learning_rate = 1e-3
#     g_learning_rate = 1e-3
#     sgd_momentum = 0.9
#
#     num_epochs = 5000
#     print_interval = 100
#     d_steps = 20
#     g_steps = 20
#
#     dfe, dre, ge = 0, 0, 0
#     d_real_data, d_fake_data, g_fake_data = None, None, None
#
#     discriminator_activation_function = torch.sigmoid
#     generator_activation_function = torch.tanh
#
#     d_sampler = get_distribution_sampler(data_mean, data_stddev)
#     gi_sampler = get_generator_input_sampler()
#
#     G = Generator(input_size=g_input_size,
#                   hidden_size=g_hidden_size,
#                   output_size=g_output_size,
#                   f=generator_activation_function)
#     D = Discriminator(input_size=d_input_func(d_input_size),
#                       hidden_size=d_hidden_size,
#                       output_size=d_output_size,
#                       f=discriminator_activation_function)
#
#     criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
#     d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
#     g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)
#
#     for epoch in range(num_epochs):
#         for d_index in range(d_steps):
#             # 1. Train D on real+fake
#             D.zero_grad()
#
#             #  1A: Train D on real
#             d_real_data = Variable(d_sampler(d_input_size))
#             d_real_decision = D(preprocess(d_real_data))
#             d_real_error = criterion(d_real_decision, Variable(torch.ones([1, 1])))  # ones = true
#             d_real_error.backward()  # compute/store gradients, but don't change params
#
#             #  1B: Train D on fake
#             d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
#             d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
#             d_fake_decision = D(preprocess(d_fake_data.t()))
#             d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1, 1])))  # zeros = fake
#             d_fake_error.backward()
#             d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
#
#             dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
#
#         for g_index in range(g_steps):
#             # 2. Train G on D's response (but DO NOT train D on these labels)
#             G.zero_grad()
#
#             gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
#             g_fake_data = G(gen_input)
#             dg_fake_decision = D(preprocess(g_fake_data.t()))
#             g_error = criterion(dg_fake_decision, Variable(torch.ones([1, 1])))  # Train G to pretend it's genuine
#
#             g_error.backward()
#             g_optimizer.step()  # Only optimizes G's parameters
#             ge = extract(g_error)[0]
#
#         if epoch % print_interval == 0:
#             print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
#                   (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))
#
#     if matplotlib_is_available:
#         print("Plotting the generated distribution...")
#         values = extract(g_fake_data)
#         print(" Values: %s" % (str(values)))
#         plt.hist(values, bins=50)
#         plt.xlabel('Value')
#         plt.ylabel('Count')
#         plt.title('Histogram of Generated Distribution')
#         plt.grid(True)
#         plt.show()
#
#
# train()


import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)  # [64, 784]
        img = img.view(img.size(0), *img_shape)
        return img  # [64, 1, 28, 28]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # [64, 1, 28, 28]
        img_flat = img.view(img.size(0), -1)  # [64, 784]
        validity = self.model(img_flat)  # [64, 1]

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(datasets.MNIST("../../data/mnist", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])]),), batch_size=opt.batch_size, shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
best_loss = np.inf
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------
        # 清空上次生成器的梯度计算值
        optimizer_G.zero_grad()
        # 随机的噪声数据作为输入 [64, 100]
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        # Generate a batch of images, 输入[64, 100] -> [64, 1, 28, 28]
        gen_imgs = generator(z)
        gen_output = discriminator(gen_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(gen_output, valid)
        # 反向传播计算参数梯度
        g_loss.backward()
        # 参数更新
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        totalLoss = d_loss.item() + g_loss.item()
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    if best_loss > totalLoss:
        best_loss = totalLoss

        state_dict_discriminator = discriminator.module.state_dict()
        state_dict_generator = generator.module.state_dict()

        save_path_generator = './checkpoints/generator_best.pth'
        save_path_discriminator = './checkpoints/discriminator_best.pth'

        for key in state_dict_generator.keys():
            state_dict_generator[key] = state_dict_generator[key].cpu()
        torch.save({
            "model": state_dict_generator,
            "epoch": epoch+1
        }, save_path_generator)

        for key in state_dict_discriminator.keys():
            state_dict_discriminator[key] = state_dict_discriminator[key].cpu()
        torch.save({
            "model": state_dict_discriminator,
            "epoch": epoch + 1
        }, save_path_discriminator)
        print('the model is saved in %s' % save_path_discriminator)




