import pprint as pp


fine_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]

mapping_coarse_fine = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road',
                                      'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain',
                                     'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee',
                                       'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
              'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


def print_fine_labels():
    for id, label in enumerate(fine_labels):
        print(id, " ", label)


def new_dicts():
    # fine label name -> id of fine label
    fine_id = dict()
    # id of fine label -> fine label name
    id_fine = dict()
    for id, label in enumerate(fine_labels):
        fine_id[label] = id
        id_fine[id] = label

    # coarse label name -> id of coarse label
    coarse_id = dict()
    # id of coarse label -> name of the coarse label
    id_coarse = dict()
    # name of fine label -> name of coarse label
    fine_coarse = dict()
    # id of fine label -> id of coarse label
    fine_id_coarse_id = dict()
    # id of coarse label -> id of fine label
    coarse_id_fine_id = dict()
    for id, (coarse, fines) in enumerate(mapping_coarse_fine.items()):
        coarse_id[coarse] = id
        id_coarse[id] = coarse
        fine_labels_ids = []
        for fine in fines:
            fine_coarse[fine] = coarse
            fine_label_id = fine_id[fine]
            fine_id_coarse_id[fine_label_id] = id
            fine_labels_ids.append(fine_label_id)
        coarse_id_fine_id[id] = fine_labels_ids

    dicts = ['fine_id', 'id_fine', 'coarse_id', 'id_coarse', 'fine_coarse',
             'fine_id_coarse_id', 'coarse_id_fine_id']
    for dic in dicts:
        dic_value = locals()[dic]
        print(dic + ' = ')
        pp.pprint(dic_value)


coarse_id_fine_id = {0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91],
                     2: [54, 62, 70, 82, 92], 3: [9, 10, 16, 28, 61],
                     4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
                     6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24],
                     8: [3, 42, 43, 88, 97], 9: [12, 17, 37, 68, 76],
                     10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
                     12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99],
                     14: [2, 11, 35, 46, 98], 15: [27, 29, 44, 78, 93],
                     16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
                     18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]}

fine_id_coarse_id = {4: 0, 30: 0, 55: 0, 72: 0, 95: 0, 1: 1, 32: 1, 67: 1, 73: 1, 91: 1, 54: 2, 62: 2, 70: 2, 82: 2,
                     92: 2, 9: 3, 10: 3, 16: 3, 28: 3, 61: 3, 0: 4, 51: 4, 53: 4, 57: 4, 83: 4, 22: 5, 39: 5, 40: 5,
                     86: 5, 87: 5, 5: 6, 20: 6, 25: 6, 84: 6, 94: 6, 6: 7, 7: 7, 14: 7, 18: 7, 24: 7, 3: 8, 42: 8,
                     43: 8, 88: 8, 97: 8, 12: 9, 17: 9, 37: 9, 68: 9, 76: 9, 23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
                     15: 11, 19: 11, 21: 11, 31: 11, 38: 11, 34: 12, 63: 12, 64: 12, 66: 12, 75: 12, 26: 13, 45: 13,
                     77: 13, 79: 13, 99: 13, 2: 14, 11: 14, 35: 14, 46: 14, 98: 14, 27: 15, 29: 15, 44: 15, 78: 15,
                     93: 15, 36: 16, 50: 16, 65: 16, 74: 16, 80: 16, 47: 17, 52: 17, 56: 17, 59: 17, 96: 17, 8: 18,
                     13: 18, 48: 18, 58: 18, 90: 18, 41: 19, 69: 19, 81: 19, 85: 19, 89: 19}

if __name__ == "__main__":
    print_fine_labels()
    pp.pprint(mapping_coarse_fine)
    new_dicts()