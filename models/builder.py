import torch
from copy import deepcopy
from .UNet_model import UNet

# ================ build net ================
def build_model(cfg, logger=None):
    model_cfg = cfg.model
    pretrained_weight = cfg.pretrained_weight
    
    # init model
    model_name = model_cfg.pop('name')
    model = eval(model_name)(**model_cfg) 

    # load pretrained weight
    if pretrained_weight is not None:
        load_weight = torch.load(pretrained_weight, weights_only=True)
        model.load_state_dict(load_weight['model_state_dict'], strict=True)
        logger.info(f'==> Load pretrained weight from {pretrained_weight}')

    return model

# ================ utils ================
def load_network(net, load_path, strict=True, param_key='model_state_dict'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'model_state_dict'.
    """
    print(
        f'Loading {net.__class__.__name__} model from {load_path}.')
    load_net = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'model_state_dict' in load_net:
            param_key = 'model_state_dict'
            print('Loading: params_ema does not exist, use model_state_dict.')
        load_net = load_net[param_key]
    print(' load net keys', load_net.keys)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)