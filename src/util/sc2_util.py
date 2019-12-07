import numpy as np
import math


def nn_action_to_sc2_action(nn_actions, ally_tags, enemy_tags, tag2unit_dict, move_dim=4):
    sc_action_list = list()
    for nn_action, ally_tag, enemy_tag in zip(nn_actions, ally_tags, enemy_tags):
        unit = tag2unit_dict[int(ally_tag)]
        if nn_action <= move_dim - 1:
            move_point = get_move_position(unit.position, nn_action)
            action = unit.move(move_point)
        elif nn_action == move_dim:
            action = unit.hold_position()
        else:
            int_enemy_tag = int(enemy_tag[nn_action - (move_dim + 1)])
            enemy_unit = tag2unit_dict[int_enemy_tag]
            action = unit.attack(enemy_unit)
        sc_action_list.append(action)
    return sc_action_list


def get_move_position(unit_position, move_dir, cardinal_points=4, radius=2):
    # when cardinal_points = 4:
    # move_dir = 0 -> RIGHT
    # move_dir = 1 -> UP
    # move_dir = 2 -> LEFT
    # move_dit = 3 -> DOWN

    theta = 2 * math.pi * float(move_dir) / cardinal_points
    delta = (math.cos(theta) * radius, math.sin(theta) * radius)
    position = unit_position + delta
    return position


def get_random_action(sc2_game_state, move_dim=4):
    sc_action_list = list()
    ally_units = sc2_game_state.units.owned
    enemy_units = sc2_game_state.units.enemy
    for ally_unit in ally_units:
        attackable_units = enemy_units.in_attack_range_of(ally_unit)
        action_index = np.random.randint(low=0, high=move_dim + 1 + len(attackable_units))
        if action_index <= move_dim - 1:  # move
            move_point = get_move_position(ally_unit.position, action_index)
            action = ally_unit.move(move_point)
        elif action_index == move_dim:  # hold
            action = ally_unit.hold_position()
        else:  # attack
            enemy_unit = attackable_units[action_index - (move_dim + 1)]
            action = ally_unit.attack(enemy_unit)
        sc_action_list.append(action)
    return sc_action_list
