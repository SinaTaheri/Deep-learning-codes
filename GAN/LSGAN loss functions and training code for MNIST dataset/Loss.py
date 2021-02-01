
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
 
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
 
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
 
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
 
    loss = None

    fake_label = Variable(torch.FloatTensor(logits_fake.shape).fill_(0.0), requires_grad=False).to(device)
    real_label = Variable(torch.FloatTensor(logits_real.shape).fill_(1.0), requires_grad=False).to(device)

    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_label, reduction='mean')
    loss_real = F.binary_cross_entropy_with_logits(logits_real, real_label, reduction='mean')

    loss = loss_fake+loss_real

    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None
    fake_label = Variable(torch.FloatTensor(logits_fake.shape).fill_(1.0), requires_grad=False).to(device)
    loss = F.binary_cross_entropy_with_logits(logits_fake, fake_label, reduction='mean')

    return loss


def ls_discriminator_loss(logits_real, logits_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
 
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
 
    loss = None
    

    fake_label = Variable(torch.FloatTensor(logits_fake.shape).fill_(0.0), requires_grad=False).to(device)
    real_label = Variable(torch.FloatTensor(logits_real.shape).fill_(1.0), requires_grad=False).to(device)

    loss_fake = (logits_fake**2).mean()
    loss_real = ((logits_real-1)**2).mean()

    loss = (loss_fake+loss_real)/2
    return loss

def ls_generator_loss(logits_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
 
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
 
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None
    fake_label = Variable(torch.FloatTensor(logits_fake.shape).fill_(0.0), requires_grad=False).to(device)
    loss = ((logits_fake-1)**2).mean()/2
    return loss    