import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils


from model import Generator, Discriminator
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main(input_size, output_size, gen_filter_size, dis_filter_size, num_epochs, beta1, lr, dataloader, device, ngpu):

    G = Generator(input_size, gen_filter_size, output_size, ngpu=ngpu)
    D = Discriminator(output_size, dis_filter_size, ngpu=ngpu)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, input_size, 1, 1, device=device)
    real_rabel = 1.
    fake_label = 0.

    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            D.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size, ), real_rabel, dtype=torch.float, device=device)

            output = D(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, input_size, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            G.zero_grad()
            label.fill_(real_rabel)
            output = D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

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

    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    device = torch.device("cuda:0" if (torch.cuda.is_avaliable() and ngpu > 0) else "cpu")

    main(input_size, output_size, gen_filter_size, dis_filter_size, num_epochs, beta1, lr, dataloader, device, ngpu)