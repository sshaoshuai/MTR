# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs


def get_ade_of_waymo(pred_trajs, gt_trajs, gt_valid_mask, calculate_steps=(5, 9, 15)) -> float:
    """Compute Average Displacement Error.

    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
    Returns:
        ade: Average Displacement Error

    """
    # assert pred_trajs.shape[2] in [1, 16, 80]
    if pred_trajs.shape[2] == 80:
        pred_trajs = pred_trajs[:, :, 4::5]
        gt_trajs = gt_trajs[:, 4::5]
        gt_valid_mask = gt_valid_mask[:, 4::5]

    ade = 0
    for cur_step in calculate_steps:
        dist_error = (pred_trajs[:, :, :cur_step+1, :] - gt_trajs[:, None, :cur_step+1, :]).norm(dim=-1)  # (batch_size, num_modes, pred_len)
        dist_error = (dist_error * gt_valid_mask[:, None, :cur_step+1].float()).sum(dim=-1) / torch.clamp_min(gt_valid_mask[:, :cur_step+1].sum(dim=-1)[:, None], min=1.0)  # (batch_size, num_modes)
        cur_ade = dist_error.min(dim=-1)[0].mean().item()

        ade += cur_ade

    ade = ade / len(calculate_steps)
    return ade


def get_ade_of_each_category(pred_trajs, gt_trajs, gt_trajs_mask, object_types, valid_type_list, post_tag='', pre_tag=''):
    """
    Args:
        pred_trajs (num_center_objects, num_modes, num_timestamps, 2): 
        gt_trajs (num_center_objects, num_timestamps, 2): 
        gt_trajs_mask (num_center_objects, num_timestamps): 
        object_types (num_center_objects): 

    Returns:
        
    """
    ret_dict = {}
    
    for cur_type in valid_type_list:
        type_mask = (object_types == cur_type)
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = -0.0
        if type_mask.sum() == 0:
            continue

        # calculate evaluataion metric
        ade = get_ade_of_waymo(
            pred_trajs=pred_trajs[type_mask, :, :, 0:2].detach(),
            gt_trajs=gt_trajs[type_mask], gt_valid_mask=gt_trajs_mask[type_mask]
        )
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = ade
    return ret_dict