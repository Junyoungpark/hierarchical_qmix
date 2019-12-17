

def victory_if_zero_enemy(stat_dict,
                          next_state_dict,
                          done,
                          victory_coeff=1.0,
                          reward_bias=0.0):
    next_units = next_state_dict['units']
    num_enemies_after = len(next_units.enemy)

    win = num_enemies_after == 0

    if done:
        if win:
            reward = 1.0 * victory_coeff
        else:
            reward = -1.0 * victory_coeff
    else:
        reward = 0.0
    reward += reward_bias

    return reward