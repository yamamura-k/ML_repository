import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from model import Generator, Discriminator
from utils import setup_logger

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main(input_size, output_size, gen_filter_size, dis_filter_size, num_epochs, beta1, lr, dataloader, device, ngpu):
    torch.manual_seed(0)

    G = Generator(input_size, gen_filter_size, output_size, ngpu=ngpu).to(device)
    D = Discriminator(output_size, dis_filter_size, ngpu=ngpu).to(device)
    
    if (device.type == 'cuda') and (ngpu > 1):
        G = nn.DataParallel(G, list(range(ngpu)))
        D = nn.DataParallel(D, list(range(ngpu)))

    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, input_size, 1, 1, device=device)
    real_rabel = 1.
    fake_label = 0.

    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []

    iters = 0

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
            if i%50 == 0:
                logger.info(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    torch.save(G.state_dict(), "./generator.pth")
    torch.save(D.state_dict(), "./discriminator.pth")
    return img_list, G_losses, D_losses

if __name__=="__main__":
    logger = setup_logger.setLevel(10)
    # Root directory for dataset
    dataroot = "../data/celeba_hq_female"

    workers = 2
    batch_size = 128
    image_size = 64

    output_size = 3
    input_size = 100
    gen_filter_size = 64
    dis_filter_size = 64

    num_epochs = 10
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
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    img_list, G_losses, D_losses = main(input_size, output_size, gen_filter_size, dis_filter_size, num_epochs, beta1, lr, dataloader, device, ngpu)
    
    #plot losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./DCGAN.png")

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    try:ani.save("./DCGAN.gif", writer="imagemagick")
    except:pass
    try:ani.save('./DCGAN.mp4', writer="ffmpeg")
    except:pass