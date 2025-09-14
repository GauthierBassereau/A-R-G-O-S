import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from source.configs import ImageDecoderTransposeConfig

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def horizontal_forward(network, x, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std is None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist
    

class ImageDecoderTranspose(nn.Module):
    def __init__(self, observation_shape=(3, 512, 512), feature_dim=1024,
                 activation=nn.ReLU, depth=64, kernel_size=5, stride=3):
        super().__init__()
        act = activation()
        C, H, W = observation_shape

        self.observation_shape = observation_shape
        self.feature_dim = feature_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride

        self.proj = nn.Conv2d(feature_dim, depth*32, kernel_size=1)
        self.network = nn.Sequential(
            act,
            nn.ConvTranspose2d(depth*32, depth*8, kernel_size, stride, padding=1),
            act,
            nn.ConvTranspose2d(depth*8, depth*4, kernel_size, stride, padding=1),
            act,
            nn.ConvTranspose2d(depth*4, depth*2, kernel_size, stride, padding=1),
            act,
            nn.ConvTranspose2d(depth*2, depth*1, kernel_size, stride, padding=1),
            act,
            nn.ConvTranspose2d(depth*1, C, kernel_size, stride, padding=1),
            nn.Upsample(size=(H, W), mode='bilinear', align_corners=False),
            # nn.Sigmoid()
        )
        self.apply(initialize_weights)
        
    @classmethod
    def from_config(cls, cfg):
        return cls(
            observation_shape=cfg.observation_shape,
            feature_dim=cfg.feature_dim,
            depth=cfg.depth,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride
        )

    def forward(self, z):
        B, N, E = z.shape
        Hp = Wp = int(N**0.5)  # -> 32 for N=1024
        assert Hp*Wp == N, "N must be a perfect square to form a grid"

        # (B, N, E) -> (B, E, Hp, Wp)
        z = z.view(B, Hp, Wp, E).permute(0, 3, 1, 2).contiguous()

        x = self.proj(z)         # (B, depth*32, Hp, Wp)
        x = self.network(x)      # (B, C, H, W)

        return x

    
if __name__ == "__main__":
    cfg = ImageDecoderTransposeConfig()
    model = ImageDecoderTranspose.from_config(cfg)
    model.to("mps")
    z = torch.randn(2, 1024, 1024).to("mps")
    img = model(z)
    print("img:", tuple(img.shape))