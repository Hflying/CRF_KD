"""Load a Dorado-exported CRF model (individual *.tensor files) for PyTorch eval."""
from __future__ import annotations

import glob
import os
from collections import OrderedDict

import toml
import torch

from bonito.util import load_symbol, set_config_defaults


def load_dorado_crf(dirname, device='cuda', half=True, chunksize=None,
                    batchsize=None, overlap=None):
    """Load Dorado R9.4.1 CRF weights into a Bonito-compatible model."""
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = set_config_defaults(config, chunksize, batchsize, overlap)
    Model = load_symbol(config, 'Model')
    model = Model(config)

    enc_sd = OrderedDict()
    for path in sorted(glob.glob(os.path.join(dirname, '*.tensor'))):
        name = os.path.basename(path).replace('.tensor', '')
        mod = torch.jit.load(path, map_location='cpu')
        enc_sd[name] = dict(mod.named_parameters())['0']
    model.encoder.load_state_dict(enc_sd)

    device = torch.device(device)
    if half:
        model = model.half()
    model = model.to(device).eval()
    return model
