from torch import nn
import torch
from pcfg_models import ResidualLayer
from resnet import resnet18
import torchvision

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)

def tensor_to_image(tensor): # 3d with num of channels first
    tensor = tensor.to('cpu')
    tensor = tensor * NORM_STD + NORM_MEAN
    image = torchvision.transforms.ToPILImage()(tensor)
    return image


class WordToImageProjector(nn.Module):
    def __init__(self, ndim=64, outdim=64):
        super(WordToImageProjector, self).__init__()
        self.composer = nn.Sequential(
            nn.Linear(ndim, ndim*4),
            ResidualLayer(ndim*4, ndim*4),
            ResidualLayer(ndim*4, ndim*4),
            nn.Linear(ndim*4, outdim),
        )

    def forward(self, x): # input is a set of word embeddings:
        x = self.composer(x)
        x = torch.sum(x, dim=-2).squeeze(1)
        return x


class CNNWordToImageProjector(torch.nn.Module):
    def __init__(self, ndim=64, outdim=64):
        super(CNNWordToImageProjector, self).__init__()
        self.cnns = torch.nn.ModuleList()
        self.cnns.append(torch.nn.Conv2d(1, 128, (1, ndim)))
        self.cnns.append(torch.nn.Conv2d(1, 128, (2, ndim),padding=(1,0)))
        self.cnns.append(torch.nn.Conv2d(1, 128, (3, ndim), padding=(1,0)))
        self.cnns.append(torch.nn.Conv2d(1, 128, (4, ndim), padding=(2,0)))
        self.cnns.append(torch.nn.Conv2d(1, 128, (5, ndim),padding=(2,0)))
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = torch.nn.Linear(128*5, 128*2)
        self.linear2 = torch.nn.Linear(128*2, outdim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        xs = []
        for mod in self.cnns:
            xs.append(self.pool(mod(x)))
        xs = torch.cat(xs, dim=1).squeeze(2).squeeze(2)
        xs = nn.functional.relu(xs)
        xs = self.linear2(self.relu(self.linear1(xs)))
        xs = torch.tanh(xs)
        # print(xs.size())
        return xs

class ImageEncoder(nn.Module):
    def __init__(self, state_dim=256):
        super(ImageEncoder, self).__init__()
        self.state_dim = state_dim
        self.encoder = resnet18(num_classes=state_dim)

    def forward(self, x):
        return self.encoder(x)

class ImageGenerator(nn.Module):
    def __init__(self, state_dim, ngf=64):
        super(ImageGenerator, self).__init__()
        self.state_dim = state_dim
        self.num_channels = 3
        self.num_feature_maps = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( state_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)