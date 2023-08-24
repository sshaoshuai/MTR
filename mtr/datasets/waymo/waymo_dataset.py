# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import os
import numpy as np
from pathlib import Path
import pickle
import torch

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]

        self.infos = self.get_all_infos(self.data_root / self.dataset_cfg.INFO_FILE[self.mode])
        self.logger.info(f'Total scenes after filters: {len(self.infos)}')

    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        ret_infos = self.create_scene_level_data(index)

        return ret_infos

    def create_scene_level_data(self, index):
        """
        Args:
            index (index):

        Returns:

        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
            info = pickle.load(f)

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            if info['map_infos']['all_polylines'].__len__() == 0:
                info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
                print(f'Warning: empty HDMap {scene_id}')

            map_polylines_data, map_polylines_mask, map_polylines_center = self.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=info['map_infos'],
                center_offset=self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
            )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

            ret_dict['map_polylines'] = map_polylines_data
            ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
            ret_dict['map_polylines_center'] = map_polylines_center

        return ret_dict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids
        ):
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future
        )

        # generate the labels of track_objects for training
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
        """[summary]

        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)

        # transform coordinates to the centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        ## generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        ##  generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask

    def create_map_data_for_center_objects(self, center_objects, map_infos, center_offset):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = torch.from_numpy(map_infos['all_polylines'].copy())
        center_objects = torch.from_numpy(center_objects)

        batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
            polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
            vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
            num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
        )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.dataset_cfg.NUM_OF_SRC_POLYLINES

        if len(batch_polylines) > num_of_src_polylines:
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(num_center_objects, 1, 2),
                angle=center_objects[:, 6]
            ).view(num_center_objects, 2)

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy()

        return map_polylines, map_polylines_mask, map_polylines_center

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
                }
                cur_scene_pred_list.append(single_pred_dict)

            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        assert len(pred_dict_list) == batch_dict['batch_size']

        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', **kwargs):
        if eval_method == 'waymo':
            from .waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    args = parser.parse_args()

    import yaml
    from easydict import EasyDict
    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)



