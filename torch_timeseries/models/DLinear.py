import torch
import torch.nn as nn
from torch_timeseries.nn.decomp import SeriesDecomp

class DLinear(nn.Module):
    """
    An enhanced DLinear with MLP layers to capture non-linearity.
    """
    def __init__(self, seq_len, pred_len, enc_in, individual:bool = False, hidden_dim=128):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        def create_mlp():
            return nn.Sequential(
                nn.Linear(self.seq_len, hidden_dim),
                nn.ReLU(),  # Add non-linearity
                nn.Linear(hidden_dim, self.pred_len)
            )

        if self.individual:
            self.MLP_Seasonal = nn.ModuleList()
            self.MLP_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.MLP_Seasonal.append(create_mlp())
                self.MLP_Trend.append(create_mlp())
        else:
            self.MLP_Seasonal = create_mlp()
            self.MLP_Trend = create_mlp()

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init[:,:,:self.pred_len])
            trend_output = torch.zeros_like(trend_init[:,:,:self.pred_len])
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.MLP_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.MLP_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.MLP_Seasonal(seasonal_init)
            trend_output = self.MLP_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1)