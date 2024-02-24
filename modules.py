import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torch.nn.utils.parametrizations import remove_parametrizations

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, causal=False, mode='SAME'):
        super(Conv, self).__init__()

        self.causal = causal
        self.mode = mode
        if self.causal and self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1)
        elif self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1) // 2
        else:
            self.padding = 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

    def remove_weight_norm(self):
        nn.utils.parametrize.remove_parametrizations(self.conv, 'weight')

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation,
                 cin_channels=None, local_conditioning=True, causal=False, mode='SAME'):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.mode = mode

        self.filter_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.parametrizations.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.parametrizations.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.parametrizations.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = F.tanh(h_filter) * F.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out)
        if self.mode == 'SAME':
            return (tensor + res) * math.sqrt(0.5), skip
        else:
            return (tensor[:, :, 1:] + res) * math.sqrt(0.5), skip

    def remove_weight_norm(self):
        self.filter_conv.remove_weight_norm()
        self.gate_conv.remove_weight_norm()
        nn.utils.parametrize.remove_parametrizations(self.res_conv, 'weight')
        nn.utils.parametrize.remove_parametrizations(self.skip_conv, 'weight')
        nn.utils.parametrize.remove_parametrizations(self.filter_conv_c, 'weight')
        nn.utils.parametrize.remove_parametrizations(self.gate_conv_c, 'weight')

class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        new_average = self.decay * x + (1.0 - self.decay) * self.shadow[name]
        self.shadow[name] = new_average.clone()


def stft(y):
    window = torch.hann_window(1024)
    if torch.cuda.is_available():
        y = y.cuda()
        window = window.cuda()
    
    D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024,
                   window=window, return_complex=True)
    
    # Ensure D is complex for sanity check
    assert D.is_complex(), "D should be a complex tensor after STFT"
    
    # Compute magnitude (result is real)
    D_magnitude = torch.sqrt(D.real**2 + D.imag**2 + 1e-10)
    
    # Sanity check to ensure D_magnitude is real
    assert not D_magnitude.is_complex(), "D_magnitude should be real"
    
    # Apply logarithmic scaling to the magnitude
    S = 2 * torch.log(torch.clamp(D_magnitude, min=1e-10))
    
    return D_magnitude, S
