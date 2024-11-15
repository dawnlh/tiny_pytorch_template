from .pix_loss import WeightedLoss

# ================ build net ================
def build_loss(cfg, logger=None):
    loss_cfg = cfg.loss
    loss_name = loss_cfg.pop('name')
    loss = eval(loss_name)(loss_cfg)    
    return loss