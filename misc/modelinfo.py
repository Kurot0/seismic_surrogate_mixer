import yaml
import argparse
import importlib
import torch
from torchinfo import summary


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', type=str)
args = parser.parse_args()

with open(args.config, 'r') as stream:
    params = yaml.safe_load(stream)

model_module = importlib.import_module(params['model_module'])
model_class = getattr(model_module, params['model_class'])

out_chans = len(params['output_channels'])
model = model_class(out_chans=out_chans, **params)

x = torch.randn(4, 6, 720, 896)
underground_data = torch.randn(1, 13, 400, 400)

summary(
    model,
    col_names=["output_size", "num_params"],
    input_data=(x, underground_data)
)