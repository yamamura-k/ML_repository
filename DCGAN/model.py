import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, filter_size, output_size, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_size, filter_size*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filter_size*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(filter_size*8, filter_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filter_size*4, filter_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filter_size*2, filter_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(filter_size, output_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, inputs):
        return self.model(inputs)


class Discriminator(nn.Module):
    def __init__(self, output_size, filter_size, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(output_size, filter_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size, filter_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size*2, filter_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(filter_size*4, filter_size*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(filter_size*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        return self.model(inputs)