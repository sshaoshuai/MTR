# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .mtr_encoder import MTREncoder

__all__ = {
    'MTREncoder': MTREncoder,
}


def build_context_encoder(config):
    model = __all__[config.NAME](
        config=config
    )

    return model
