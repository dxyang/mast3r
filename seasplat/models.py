import torch
import torch.nn as nn

class BackscatterNet(nn.Module):
    '''
    backscatter = B_inf * (1 - exp(- a * z)) + J_prime * exp(- b * z)
    '''
    def __init__(self, use_residual: bool = False, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()

        self.scale = scale

        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

        self.use_residual = use_residual
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))

        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()

        self.do_sigmoid = do_sigmoid

    def forward(self, depth):
        if self.do_sigmoid:
            beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params)))
        else:
            beta_b_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.backscatter_conv_params), 0.0)

        backscatter = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params)))
            else:
                beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.residual_conv_params), 0.0)
            backscatter += torch.sigmoid(self.J_prime) * torch.exp(-beta_d_conv)

        return backscatter

class AttenuateNet(nn.Module):
    '''
    attenuation_map = exp(-beta_d * z)
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()

        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
        self.attenuation_coef = None
        self.scale = scale
        self.do_sigmoid = do_sigmoid

        self.relu = nn.ReLU()

    def forward(self, depth):
        if self.do_sigmoid:
            beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params)))
        else:
            beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0)

        attenuation_map = torch.exp(-beta_d_conv)

        return attenuation_map