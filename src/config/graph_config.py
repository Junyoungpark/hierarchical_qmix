# Node relations
NODE_ALLY = 0
NODE_ENEMY = 1

# Edge relationships
EDGE_ALLY = 0
EDGE_ENEMY = 1
EDGE_ALLY_TO_ENEMY = 2
EDGE_IN_ATTACK_RANGE = 3

# whenever you edit 'node relationships', 'edge relationships' please make sure to update
# 'NUM_NODE_TYPES', 'NUM_EDGE_TYPES' variable
# Todo: Programmatically managing 'NUM_NODE_TYPES' variable
# Todo: Programmatically managing 'NUM_EDGE_TYPES' variable

NUM_NODE_TYPES = 2
NUM_EDGE_TYPES = 4

# Edge key
EDGE_ALLIES_KEY = 0
EDGE_ENEMY_KEY = 1
EDGE_IN_ATTACK_RANGE_KEY = 2
EDGE_IN_ENEMY_ATTACK_RANGE_KEY = 3
