import torch
import torch.nn as nn



class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        '''
        self.fc1 = MLP(num_channels, num_channels_reduced, num_channels)
        self.fc2 = MLP(num_channels, num_channels_reduced, num_channels)
        '''
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialAttention(nn.Module):
     def __init__(self, kernel_size=7):
         super(SpatialAttention, self).__init__()
         # 'kernel size must be 3 or 7'
         assert kernel_size in (3, 7)
         padding = 3 if kernel_size == 7 else 1

         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
         self.sigmoid = nn.Sigmoid()

     def forward(self, x):
         avg_out = torch.mean(x, dim=1, keepdim=True)
         max_out, _ = torch.max(x, dim=1, keepdim=True)
         x = torch.cat([avg_out, max_out], dim=1)
         x = self.conv(x)
         return self.sigmoid(x)
class SSEAttention(nn.Module):
     # CSP Bottleneck with 3 convolutions
     def __init__(self, channel):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SSEAttention, self).__init__()
        '''
        # c_ = int(c2 * e)  # hidden channels
         # self.cv1 = Conv(c1, c_, 1, 1)
         # self.cv2 = Conv(c1, c_, 1, 1)
         # self.cv3 = Conv(2 * c_, c2, 1)
         # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
         '''
        self.channel_attention = SELayer(channel, 8)
        self.spatial_attention = SpatialAttention(7)

         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

     def forward(self, x):
         out = self.channel_attention(x) * x
         # print('outchannels:{}'.format(out.shape))
         out = self.spatial_attention(out) * out
         return out
