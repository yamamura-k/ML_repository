import torch
import torch.optim as optim
import torch.nn as nn


from model import Generator, Discriminator
def main(dataloader, ngpu, beta1, lr, num_epochs, input_size, output_size, gen_filter_size, dis_filter_size):

    G = Generator()
    D = Discriminator()
if __name__=="__main__":
    # Root directory for dataset
    dataroot = "data/celeba"

    workers = 2
    batch_size = 128
    image_size = 64

    output_size = 3
    input_size = 100
    gen_filter_size = 64
    dis_filter_size = 64

    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1