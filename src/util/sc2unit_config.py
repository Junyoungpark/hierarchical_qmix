# Helper dictionary for encoding unit types: python-sc2 unit_typeid -> one-hot position

type2onehot = {9: 0,  # Baneling
               105: 1,  # Zergling
               48: 2,  # Marine
               51: 3,  # Marauder
               33: 4,  # Siegetank - tank mode
               32: 5,  # Siegetank - siege mode
               54: 6,  # Medivac
               73: 7,  # Zealot
               53: 8  # Hellion
               }

NUM_TOTAL_TYPES = len(type2onehot)