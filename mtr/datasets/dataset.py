# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.utils.data as torch_data
import mtr.utils.common_utils as common_utils


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.logger = logger

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def collate_batch(self, batch_list):
        """
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
            map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos: (num_center_objects, num_objects, 3)
            obj_types: (num_objects)
            obj_ids: (num_objects)

            center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)

            obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
            obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
            center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
            center_gt_trajs_mask (num_center_objects, num_future_timestamps):
            center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
        """
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():

            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
        return batch_dict
