import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_timeseries.norm_experiments.experiment import NormExperiment
import torch.nn as nn
from dataclasses import asdict, dataclass
from torch_timeseries.models import Fredformer

from dataclasses import dataclass, asdict

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class FredformerExperiment(NormExperiment):
    model_type: str = "Fredformer"
    individual: bool = False
    d_ff: int = 256
    dropout: float = 0.2
    # 沒找到哪厘調用
    fc_dropout: float = 0.2
    head_dropout: float = 0
    cf_dim: int = 32
    cf_depth: int = 3
    cf_heads: int = 8
    cf_mlp: int = 32
    cf_head_dim: int = 8
    d_model: int = 8
    cf_drop: float = 0.0
    mlp_hidden: int = 0
    mlp_drop: float = 0.0

    def _init_f_model(self):
        # 根据 pred_len 动态调整参数
        if self.pred_len == 720 or self.pred_len == 336:
            cf_dim = 16
            d_model = 4
            cf_depth = self.cf_depth
            cf_heads = self.cf_heads
            cf_mlp = self.cf_mlp
            cf_head_dim = self.cf_head_dim
        else:
            # 其他情况使用默认值
            cf_dim = self.cf_dim
            cf_depth = self.cf_depth
            cf_heads = self.cf_heads
            cf_mlp = self.cf_mlp
            cf_head_dim = self.cf_head_dim
            d_model = self.d_model

        configs = {
            "output": 0,
            "c_in": self.dataset.num_features,
            "context_window": self.windows,
            "target_window": self.pred_len,
            "e_layers": 2,
            "n_heads": 16,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "fc_dropout": self.fc_dropout,
            "head_dropout": self.head_dropout,
            "patch_len": 16,
            "stride": 16,
            "patience": 10,
            "individual": 0,
            "padding_patch": None,
            "affine": None,
            "subtract_last": None,
            "use_nys": 0,
            "ablation": None,
            "revin": 0,
            "cf_dim": cf_dim,
            "cf_depth": cf_depth,
            "cf_heads": cf_heads,
            "cf_mlp": cf_mlp,
            "cf_head_dim": cf_head_dim,
            "d_model": d_model,
            "cf_drop": self.cf_drop,
            "mlp_hidden": self.mlp_hidden,
            "mlp_drop": self.mlp_drop,
        }

        # 直接解包字典传递参数
        self.f_model = Fredformer(**configs)
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # x: [Batch, Input length, Channel]
        batch_x, _ = self.model.normalize(batch_x)
        batch_x = batch_x.permute(0, 2, 1)
        pred = self.model.fm(batch_x)
        pred = pred.permute(0, 2, 1)
        pred = self.model.denormalize(pred)
        return pred,batch_y

def run_experiment(pred_len):
    exp = FredformerExperiment(
        dataset_type="Weather",
        data_path="./data",
        norm_type='FAN',  # No RevIN DishTS SAN
        optm_type="Adam",
        batch_size=128,
        device="cuda:0",
        windows=96,
        pred_len=pred_len,
        horizon=1,
        epochs=100,
        # norm_config={"freq_topk": 2}
    )
    exp.run()

if __name__ == "__main__":
    pred_lens = [96, 192, 336, 720]
    # pred_lens = [96, 192, 336, 720]
    for pred_len in pred_lens:
        run_experiment(pred_len)
    print("Done")