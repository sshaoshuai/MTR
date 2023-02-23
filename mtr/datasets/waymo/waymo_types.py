# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


object_type = {
    0: 'TYPE_UNSET',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_CYCLIST',
    4: 'TYPE_OTHER'
}

lane_type = {
    0: 'TYPE_UNDEFINED',
    1: 'TYPE_FREEWAY',
    2: 'TYPE_SURFACE_STREET',
    3: 'TYPE_BIKE_LANE'
}

road_line_type = {
    0: 'TYPE_UNKNOWN',
    1: 'TYPE_BROKEN_SINGLE_WHITE',
    2: 'TYPE_SOLID_SINGLE_WHITE',
    3: 'TYPE_SOLID_DOUBLE_WHITE',
    4: 'TYPE_BROKEN_SINGLE_YELLOW',
    5: 'TYPE_BROKEN_DOUBLE_YELLOW',
    6: 'TYPE_SOLID_SINGLE_YELLOW',
    7: 'TYPE_SOLID_DOUBLE_YELLOW',
    8: 'TYPE_PASSING_DOUBLE_YELLOW'
}

road_edge_type = {
    0: 'TYPE_UNKNOWN',
    # // Physical road boundary that doesn't have traffic on the other side (e.g.,
    # // a curb or the k-rail on the right side of a freeway).
    1: 'TYPE_ROAD_EDGE_BOUNDARY',
    # // Physical road boundary that separates the car from other traffic
    # // (e.g. a k-rail or an island).
    2: 'TYPE_ROAD_EDGE_MEDIAN'
}

polyline_type = {
    # for lane
    'TYPE_UNDEFINED': -1,
    'TYPE_FREEWAY': 1,
    'TYPE_SURFACE_STREET': 2,
    'TYPE_BIKE_LANE': 3,

    # for roadline
    'TYPE_UNKNOWN': -1,
    'TYPE_BROKEN_SINGLE_WHITE': 6,
    'TYPE_SOLID_SINGLE_WHITE': 7,
    'TYPE_SOLID_DOUBLE_WHITE': 8,
    'TYPE_BROKEN_SINGLE_YELLOW': 9,
    'TYPE_BROKEN_DOUBLE_YELLOW': 10,
    'TYPE_SOLID_SINGLE_YELLOW': 11,
    'TYPE_SOLID_DOUBLE_YELLOW': 12,
    'TYPE_PASSING_DOUBLE_YELLOW': 13,

    # for roadedge
    'TYPE_ROAD_EDGE_BOUNDARY': 15,
    'TYPE_ROAD_EDGE_MEDIAN': 16,

    # for stopsign
    'TYPE_STOP_SIGN': 17,

    # for crosswalk
    'TYPE_CROSSWALK': 18,

    # for speed bump
    'TYPE_SPEED_BUMP': 19
}


signal_state = {
    0: 'LANE_STATE_UNKNOWN',

    # // States for traffic signals with arrows.
    1: 'LANE_STATE_ARROW_STOP',
    2: 'LANE_STATE_ARROW_CAUTION',
    3: 'LANE_STATE_ARROW_GO',

    # // Standard round traffic signals.
    4: 'LANE_STATE_STOP',
    5: 'LANE_STATE_CAUTION',
    6: 'LANE_STATE_GO',

    # // Flashing light signals.
    7: 'LANE_STATE_FLASHING_STOP',
    8: 'LANE_STATE_FLASHING_CAUTION'
}

signal_state_to_id = {}
for key, val in signal_state.items():
    signal_state_to_id[val] = key