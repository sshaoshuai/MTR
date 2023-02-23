# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import tensorflow as tf
import os

from google.protobuf import text_format

all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, True)
    except RuntimeError as e:
        print(e)

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2


object_type_to_id = {
    'TYPE_UNSET': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_CYCLIST': 3,
    'TYPE_OTHER': 4
}


def _default_metrics_config(eval_second, num_modes_for_eval=6):
    assert eval_second in [3, 5, 8]
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
    }
    """
    config_text += f"""
    max_predictions: {num_modes_for_eval}
    """
    if eval_second == 3:
        config_text += """
        track_future_samples: 30
        """
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config


def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f'Total number for evaluation (after processed): {len(pred_dicts)}')

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if  cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    if num_future_frames in [30, 50, 80]:
        sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    num_frame_to_eval = num_future_frames // sampled_interval

    if eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6
    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10
    else:
        num_frames_in_total = 91
        num_frame_to_eval = 16

    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)

    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]

            cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()

            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict



def waymo_evaluation(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6):

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = transform_preds_to_waymo_format(
        pred_dicts, top_k_for_eval=top_k, eval_second=eval_second,
    )
    eval_config = _default_metrics_config(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval)

    pred_score = tf.convert_to_tensor(pred_score, np.float32)
    pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
    gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
    gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], np.bool)
    pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], np.bool)
    object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)

    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
        prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, num_agents_per_group)
        object_type=object_type  # (batch_size, num_total_agents)
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']):
        avg_results.update({
            f'{m} - VEHICLE': [0.0, 0], f'{m} - PEDESTRIAN': [0.0, 0], f'{m} - CYCLIST': [0.0, 0]
        })
        for j, n in enumerate(metric_names):
            cur_name = n.split('_')[1]
            avg_results[f'{m} - {cur_name}'][0] += float(metric_results[i][j])
            avg_results[f'{m} - {cur_name}'][1] += 1
            result_dict[f'{m} - {n}\t'] = float(metric_results[i][j])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / avg_results[key][1]

    result_dict['-------------------------------------------------------------'] = 0
    result_dict.update(avg_results)

    final_avg_results = {}
    result_format_list = [
        ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', '\n'],
        ['VEHICLE', None, None, None, None, '\n'],
        ['PEDESTRIAN', None, None, None, None, '\n'],
        ['CYCLIST', None, None, None, None, '\n'],
        ['Avg', None, None, None, None, '\n'],
    ]
    name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
    name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4}

    for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP']:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name} - {cur_name}']

            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = '%.4f,' % avg_results[f'{cur_metric_name} - {cur_name}']

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

    result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])

    result_dict['--------------------------------------------------------------'] = 0
    result_dict.update(final_avg_results)
    result_dict['---------------------------------------------------------------'] = 0
    result_dict.update(object_type_cnt_dict)
    result_dict['-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'] = 0

    return result_dict, result_format_str


def main():
    import pickle
    import argparse
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--top_k', type=int, default=-1, help='')
    parser.add_argument('--eval_second', type=int, default=8, help='')
    parser.add_argument('--num_modes_for_eval', type=int, default=6, help='')

    args = parser.parse_args()
    print(args)

    assert args.eval_second in [3, 5, 8]
    pred_infos = pickle.load(open(args.pred_infos, 'rb'))

    result_format_str = ''
    print('Start to evaluate the waymo format results...')

    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=pred_infos, top_k=args.top_k, eval_second=args.eval_second,
        num_modes_for_eval=args.num_modes_for_eval,
    )

    print(metric_results)
    metric_result_str = '\n'
    for key in metric_results:
        metric_results[key] = metric_results[key]
        metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
    print(metric_result_str)
    print(result_format_str)


if __name__ == '__main__':
    main()

