import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange

# Define a model registry
model_registry = {}


# Create a decorator to register models
def register_model(model_name):
    def decorator(cls):
        model_registry[model_name] = cls
        return cls

    return decorator


# Define a function to retrieve and instantiate the model class by model_name
def get_model_by_name(model_name, *args, **kwargs):
    model_cls = model_registry.get(model_name)
    if model_cls is None:
        raise ValueError(f"No model found with model_name{model_name}.")
    return model_cls(*args, **kwargs)


# Use the decorator to register the model class

@register_model('FANLayer')
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


@register_model('FANLayerGated')
class FANLayerGated(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, gated=True):
        super(FANLayerGated, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        return output

# 123
class FAN3_MLP_DualStream(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                 freq_dim=256, time_dim=256,
                 hidden_dim=512, num_heads=4):
        super().__init__()

        # 双流编码器
        self.freq_stream = nn.Sequential(
            nn.Linear(input_dim, freq_dim),
            nn.LayerNorm(freq_dim)
        )

        self.time_stream = nn.Sequential(
            nn.Linear(input_dim, time_dim),
            nn.LayerNorm(time_dim)
        )
        # 交叉注意力模块
        self.cross_attn = CrossAttentionBlock(
            dim=freq_dim,
            context_dim=time_dim,
            num_heads=num_heads
        )

        # 融合预测头
        self.head = nn.Sequential(
            nn.Linear(freq_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, main_freq, x_time):
        # 独立编码
        freq_feat = self.freq_stream(main_freq)  # [B, L, freq_dim]
        time_feat = self.time_stream(x_time)  # [B, L, time_dim]

        # 交叉注意力
        fused_feat = self.cross_attn(
            query=freq_feat,
            context=time_feat
        )  # [B, L, freq_dim]

        # 特征拼接
        combined = torch.cat([fused_feat, time_feat], dim=-1)

        # 最终预测
        return self.head(combined)
class FAN2_DualStream(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                 freq_dim=256, time_dim=256,
                 hidden_dim=512, num_heads=4):
        super().__init__()

        # 双流编码器
        self.freq_stream = nn.Sequential(
            FANLayer(input_dim, freq_dim),
            nn.LayerNorm(freq_dim)
        )

        self.time_stream = nn.Sequential(
            FANLayer(input_dim, time_dim),
            nn.LayerNorm(time_dim)
        )
        # 交叉注意力模块
        self.cross_attn = CrossAttentionBlock(
            dim=freq_dim,
            context_dim=time_dim,
            num_heads=num_heads
        )

        # 融合预测头
        self.head = nn.Sequential(
            nn.Linear(freq_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, main_freq, x_time):
        # 独立编码
        freq_feat = self.freq_stream(main_freq)  # [B, L, freq_dim]
        time_feat = self.time_stream(x_time)  # [B, L, time_dim]

        # 交叉注意力
        fused_feat = self.cross_attn(
            query=freq_feat,
            context=time_feat
        )  # [B, L, freq_dim]

        # 特征拼接
        combined = torch.cat([fused_feat, time_feat], dim=-1)

        # 最终预测
        return self.head(combined)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query来自主特征(freq)
        self.to_q = nn.Linear(dim, dim, bias=False)
        # Key/Value来自上下文特征(time)
        self.to_kv = nn.Linear(context_dim, 2 * dim, bias=False)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, context):
        """
        query:  频域特征 [B, L, D]
        context: 时域特征 [B, L, C]
        """
        B, L, _ = query.shape

        # 1. 投影到Q/K/V
        q = self.to_q(query)  # [B, L, D]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [B, L, D], [B, L, D]

        # 2. 多头切分
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # 3. 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 4. 特征聚合
        out = attn @ v  # [B, h, L, d]
        out = rearrange(out, 'b h l d -> b l (h d)')

        # 5. 残差连接
        return self.norm(query + self.proj(out))
# 123
@register_model('FAN2')
class FAN2(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=512, num_layers=3):
        super(FAN2, self).__init__()
        self.model_freq = FANLayer(input_dim, hidden_dim // 2)
        self.model_all = nn.Sequential(
            FANLayer(hidden_dim // 2 + input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)


@register_model('FANGated')
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated=True):
        super(FANGated, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated=gated))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output


@register_model('MLP')
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        self.activation = nn.GELU()
        self.layers = nn.ModuleList()
        if use_embedding:
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        else:
            self.layers.extend([nn.Linear(input_dim, hidden_dim), self.activation])

        for _ in range(num_layers - 2):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src) if hasattr(self, 'embedding') else src
        for layer in self.layers:
            output = layer(output)
        return output


class RoPEPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


@register_model('Transformer')
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=768, num_layers=12, num_heads=12, norm_first=True,
                 encoder_only=True, decoder_only=False):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = RoPEPositionalEncoding(hidden_dim)
        self.encoder_only = encoder_only
        self.decoder_only = decoder_only
        assert not (self.encoder_only and self.decoder_only)
        if self.encoder_only:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first=norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        elif self.decoder_only:
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first=norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        else:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first=norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers // 2)
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first=norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers // 2)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src).unsqueeze(0)
        src = self.pos_encoder(src)
        if self.encoder_only:
            src = self.transformer_encoder(src)
        elif self.decoder_only:
            src = self.transformer_decoder(src, src)
        else:
            src = self.transformer_encoder(src)
            src = self.transformer_decoder(src, src)
        output = self.out(src)
        return output


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )

        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


@register_model('KAN')
class KAN(nn.Module):
    def __init__(
            self,
            input_dim=1,
            output_dim=1,
            hidden_dim=128,
            num_layers=3,
            grid_size=50,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        layers_hidden = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
