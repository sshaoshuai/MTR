# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
from torch.utils.data import DataLoader
from mtr.utils import common_utils

from .waymo.waymo_dataset import WaymoDataset


__all__ = {
    'WaymoDataset': WaymoDataset,
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, add_worker_init_fn=False):
    
    def worker_init_fn_(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        training=training,
        logger=logger, 
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    drop_last = dataset_cfg.get('DATALOADER_DROP_LAST', False) and training
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0, 
        worker_init_fn=worker_init_fn_ if add_worker_init_fn and training else None
    )

    return dataset, dataloader, sampler
