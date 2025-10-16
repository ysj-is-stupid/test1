
import time
import torch
import torch.nn as nn


def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim = 1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    # Xres = Xt −Xnon
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered



class FAN(nn.Module):
    """FAN first substract bottom k frequecy component from the original series


    Args:
        nn (_type_): _description_
    """
    def __init__(self,  seq_len, pred_len, enc_in, freq_topk = 3, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk )
        self.rfft = rfft

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        # self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):
        # freq normalization
        B , O, N= true.shape
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)

        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, len, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4,
                                        bias=bias)  # There is almost no difference between bias and non-bias in our experiments.
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output
class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            FANLayer(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)


# ==================== START: 替换为下面的 V2 版本 ====================

class KalmanPredictorV2(nn.Module):
    """
    改进版的卡尔曼思想预测器
    - 增加模型容量 (state_dim, num_layers)
    - 改进状态初始化方式 (RNN Encoder)
    - 增加激活函数
    """

    def __init__(self, seq_len, pred_len, enc_in):
        super(KalmanPredictorV2, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # 1. 提升模型容量
        self.state_dim = 32  # 将状态维度从4增加到32
        self.rnn_layers = 2  # 使用2层RNN增加深度

        # 编码器：使用RNN来处理整个输入序列，获取更丰富的历史状态
        self.state_encoder_rnn = nn.RNN(
            input_size=self.enc_in,
            hidden_size=self.state_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=0.1
        )

        # 状态转移模型：同样使用RNN来预测未来状态的演化
        self.transition_model = nn.RNN(
            input_size=self.state_dim,
            hidden_size=self.state_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=0.1
        )

        # 观测模型：从潜在状态映射到观测值，增加非线性激活
        self.observation_model = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 2),
            nn.GELU(),
            nn.Linear(self.state_dim * 2, self.enc_in)
        )

    def forward(self, x_filtered):
        # x_filtered shape: (Batch, seq_len, enc_in)

        # 1. 改进的状态初始化：RNN完整编码输入序列
        # state_encoder_rnn 的输出是 (outputs, h_n)
        # 我们需要最后的隐藏状态 h_n
        _, h_n = self.state_encoder_rnn(x_filtered)
        # h_n 的形状是 (num_layers, B, state_dim)

        # 2. 迭代预测未来状态
        # 准备一个全零的输入序列，让 transition_model 自主演化
        future_inputs = torch.zeros(x_filtered.size(0), self.pred_len, self.state_dim).to(x_filtered.device)

        # 使用编码器得到的最终隐藏状态 h_n作为解码器（转移模型）的初始隐藏状态
        all_future_states, _ = self.transition_model(future_inputs, h_n)
        # all_future_states shape: (B, pred_len, state_dim)

        # 3. 将未来状态映射回观测空间
        future_predictions = self.observation_model(all_future_states)  # (B, pred_len, enc_in)

        return future_predictions.permute(0, 2, 1)  # (B, enc_in, pred_len)


class HybridPredictorV2(nn.Module):
    """
    带稳定化残差连接的混合预测器
    """

    def __init__(self, seq_len, pred_len, enc_in):
        super(HybridPredictorV2, self).__init__()
        self.mlp_freq = MLPfreq(seq_len, pred_len, enc_in)
        self.kalman_predictor = KalmanPredictorV2(seq_len, pred_len, enc_in)

        # 3. 稳定残差学习：引入可学习的缩放因子 alpha
        # 初始化为0，使得训练初期Kalman分支的输出为0，不干扰主模型
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, main_freq, x):
        # main_freq 是 x_filtered, x 是原始 input
        # main_freq 和 x 的形状都是 (B, enc_in, seq_len)

        # 1. 计算基线预测
        base_prediction = self.mlp_freq(main_freq, x)

        # 2. 计算残差预测
        # KalmanPredictor 的输入需要是 (B, seq_len, enc_in)
        residual_prediction = self.kalman_predictor(main_freq.transpose(1, 2))

        # 3. 通过 alpha 进行缩放后相加
        final_prediction = base_prediction + self.alpha * residual_prediction

        return final_prediction

# ==================== END: 替换 V2 版本 ====================