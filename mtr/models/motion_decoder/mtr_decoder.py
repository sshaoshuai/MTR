# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtr.models.utils.transformer import transformer_decoder_layer
from mtr.models.utils.transformer import position_encoding_utils
from mtr.models.utils import common_layers
from mtr.utils import common_utils, loss_utils, motion_utils
from mtr.config import cfg


class MTRDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=False
        )

        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True
        )
        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)
        else:
            self.map_query_content_mlps = self.map_query_embed_mlps = None

        # define the dense future prediction layers
        self.build_dense_future_prediction_layers(
            hidden_dim=self.d_model, num_future_frames=self.num_future_frames
        )

        # define the motion query
        self.intention_points, self.intention_query, self.intention_query_mlps = self.build_motion_query(
            self.d_model, use_place_holder=self.use_place_holder
        )

        # define the motion head
        temp_layer = common_layers.build_mlps(c_in=self.d_model * 2 + map_d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers
        )

        self.forward_ret_dict = {}

    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):
        self.obj_pos_encoding_layer = common_layers.build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.dense_future_head = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )

        self.future_traj_mlps = common_layers.build_mlps(
            c_in=4 * self.num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )

    def build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers

    def build_motion_query(self, d_model, use_place_holder=False):
        intention_points = intention_query = intention_query_mlps = None

        if use_place_holder:
            raise NotImplementedError
        else:
            intention_points_file = cfg.ROOT_DIR / self.model_cfg.INTENTION_POINTS_FILE
            with open(intention_points_file, 'rb') as f:
                intention_points_dict = pickle.load(f)

            intention_points = {}
            for cur_type in self.object_type:
                cur_intention_points = intention_points_dict[cur_type]
                cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
                intention_points[cur_type] = cur_intention_points

            intention_query_mlps = common_layers.build_mlps(
                c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
            )
        return intention_points, intention_query, intention_query_mlps

    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):
        motion_reg_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True
        )
        motion_cls_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        motion_vel_heads = None 
        return motion_reg_heads, motion_cls_heads, motion_vel_heads

    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        num_center_objects, num_objects, _ = obj_feature.shape

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], self.num_future_frames, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2)  # (num_valid_objects, C)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, self.num_future_frames, 7)
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid
        self.forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

        return ret_obj_feature, ret_pred_dense_future_trajs

    def get_motion_query(self, center_objects_type):
        num_center_objects = len(center_objects_type)
        if self.use_place_holder:
            raise NotImplementedError
        else:
            intention_points = torch.stack([
                self.intention_points[center_objects_type[obj_idx]]
                for obj_idx in range(num_center_objects)], dim=0)
            intention_points = intention_points.permute(1, 0, 2)  # (num_query, num_center_objects, 2)

            intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)
            intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(-1, num_center_objects, self.d_model)  # (num_query, num_center_objects, C)
        return intention_query, intention_points

    def apply_cross_attention(self, kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer,
                              dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                              query_content_pre_mlp=None, query_embed_pre_mlp=None):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape
        searching_query = position_encoding_utils.gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = position_encoding_utils.gen_sineembed_for_position(kv_pos, hidden_dim=d_model)

        if not use_local_attn:
            query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature.permute(1, 0, 2),
                memory_key_padding_mask=~kv_mask,
                pos=kv_pos_embed,
                is_first=(layer_idx == 0)
            )  # (M, B, C)
        else:
            batch_size, num_kv, _ = kv_feature.shape

            kv_feature_stack = kv_feature.flatten(start_dim=0, end_dim=1)
            kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
            kv_mask_stack = kv_mask.view(-1)

            key_batch_cnt = num_kv * torch.ones(batch_size).int().to(kv_feature.device)
            query_index_pair = query_index_pair.view(batch_size * num_q, -1)
            index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(-1)  # (batch_size * num_q)
            assert len(query_index_pair) == len(index_pair_batch)

            query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature_stack,
                memory_valid_mask=kv_mask_stack,
                pos=kv_pos_embed_stack,
                is_first=(layer_idx == 0),
                key_batch_cnt=key_batch_cnt,
                index_pair=query_index_pair,
                index_pair_batch=index_pair_batch
            )
            query_feature = query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2)  # (M, B, C)

        return query_feature

    def apply_dynamic_map_collection(self, map_pos, map_mask, pred_waypoints, base_region_offset, num_query, num_waypoint_polylines=128, num_base_polylines=256, base_map_idxs=None):
        map_pos = map_pos.clone()
        map_pos[~map_mask] = 10000000.0
        num_polylines = map_pos.shape[1]

        if base_map_idxs is None:
            base_points = torch.tensor(base_region_offset).type_as(map_pos)
            base_dist = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            base_topk_dist, base_map_idxs = base_dist.topk(k=min(num_polylines, num_base_polylines), dim=-1, largest=False)  # (num_center_objects, topk)
            base_map_idxs[base_topk_dist > 10000000] = -1
            base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)  # (num_center_objects, num_query, num_base_polylines)
            if base_map_idxs.shape[-1] < num_base_polylines:
                base_map_idxs = F.pad(base_map_idxs, pad=(0, num_base_polylines - base_map_idxs.shape[-1]), mode='constant', value=-1)

        dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1)  # (num_center_objects, num_query, num_polylines, num_timestamps)
        dynamic_dist = dynamic_dist.min(dim=-1)[0]  # (num_center_objects, num_query, num_polylines)

        dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(k=min(num_polylines, num_waypoint_polylines), dim=-1, largest=False)
        dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1
        if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:
            dynamic_map_idxs = F.pad(dynamic_map_idxs, pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]), mode='constant', value=-1)

        collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)  # (num_center_objects, num_query, num_collected_polylines)

        # remove duplicate indices
        sorted_idxs = collected_idxs.sort(dim=-1)[0]
        duplicate_mask_slice = (sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0)  # (num_center_objects, num_query, num_collected_polylines - 1)
        duplicate_mask = torch.ones_like(collected_idxs).bool()
        duplicate_mask[..., 1:] = duplicate_mask_slice
        sorted_idxs[~duplicate_mask] = -1

        return sorted_idxs.int(), base_map_idxs

    def apply_transformer_decoder(self, center_objects_feature, center_objects_type, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos):
        intention_query, intention_points = self.get_motion_query(center_objects_type)
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  # (num_center_objects, num_query, 2)

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]

        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  # (num_query, num_center_objects, C)

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
        dynamic_query_center = intention_points

        pred_list = []
        for layer_idx in range(self.num_decoder_layers):
            # query object feature
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
                query_content=query_content, query_embed=intention_query,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx
            ) 

            # query map feature
            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query
            )

            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                query_content=query_content, query_embed=intention_query,
                attention_layer=self.map_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=True,
                query_index_pair=collected_idxs,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps
            ) 

            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            pred_list.append([pred_scores, pred_trajs])

            # update
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2)  # (num_query, num_center_objects, 2)

        if self.use_place_holder:
            raise NotImplementedError

        assert len(pred_list) == self.num_decoder_layers
        return pred_list

    def get_decoder_loss(self, tb_pre_tag=''):
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']
        intention_points = self.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # (num_center_objects, 2)

        if not self.use_place_holder:
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
        else:
            raise NotImplementedError

        tb_dict = {}
        disp_dict = {}
        total_loss = 0
        for layer_idx in range(self.num_decoder_layers):
            if self.use_place_holder:
                raise NotImplementedError

            pred_scores, pred_trajs = pred_list[layer_idx]
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            loss_cls = F.cross_entropy(input=pred_scores, target=center_gt_positive_idx, reduction='none')

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel + loss_cls.sum(dim=-1) * weight_cls
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls

            if layer_idx + 1 == self.num_decoder_layers:
                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=tb_pre_tag
                )
                tb_dict.update(layer_tb_dict_ade)
                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / self.num_decoder_layers
        return total_loss, tb_dict, disp_dict

    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):
        obj_trajs_future_state = self.forward_ret_dict['obj_trajs_future_state'].cuda()
        obj_trajs_future_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()
        pred_dense_trajs = self.forward_ret_dict['pred_dense_trajs']  # (num_center_objects, num_objects, num_future_frames, 7)
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
        loss_reg = loss_reg.mean()

        if tb_dict is None:
            tb_dict = {}
        if disp_dict is None:
            disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        return loss_reg, tb_dict, disp_dict

    def get_loss(self, tb_pre_tag=''):
        loss_decoder, tb_dict, disp_dict = self.get_decoder_loss(tb_pre_tag=tb_pre_tag)
        loss_dense_prediction, tb_dict, disp_dict = self.get_dense_future_prediction_loss(tb_pre_tag=tb_pre_tag, tb_dict=tb_dict, disp_dict=disp_dict)

        total_loss = loss_decoder + loss_dense_prediction
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()
        disp_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, pred_list, batch_dict):
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.softmax(pred_scores, dim=-1)  # (num_center_objects, num_query)

        num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape
        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes
            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.batch_nms(
                pred_trajs=pred_trajs, pred_scores=pred_scores,
                dist_thresh=self.model_cfg.NMS_DIST_THRESH,
                num_ret_modes=self.num_motion_modes
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        return pred_scores_final, pred_trajs_final

    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # input projection
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # dense future prediction
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )
        # decoder layers
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos
        )

        self.forward_ret_dict['pred_list'] = pred_list

        if not self.training:
            pred_scores, pred_trajs = self.generate_final_prediction(pred_list=pred_list, batch_dict=batch_dict)
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs

        else:
            self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
            self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
            self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
            self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
            self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']

            self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

        return batch_dict
