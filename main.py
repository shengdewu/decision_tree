import numpy as np
from dt_id3 import utility
import load_data

if __name__ == '__main__':
    print('start excute')

    tool = utility()
    data_set = [
        [1, 1, 0,'yes'],
        [1, 1, 1,'yes'],
        [1, 0, 2,'no'],
        [0, 1, 3,'no'],
        [0, 1, 4,'no']
    ]
    # ent = tool.calc_entropy(data_set)
    #
    # print(str('the data_set {} \n entroy is {}\n').format(data_set, ent))
    #
    # ret = tool.split_data(data_set, 0, 1)
    #
    # print(str('the data_set {} \n splite by 0-1 is {}\n').format(data_set, ret))

    # best_feature = tool.calc_best_ft(data_set)
    #
    # print(str('the data_set {} \n best split {}\n').format(data_set, best_feature))
    #
    feature_label = ['no surfacing', 'flippers', 'number']
    tree = tool.create_tree(data_set, feature_label)

    print(str('the data_set {} \n tree {}\n').format(data_set, tree))

    feature_label = ['no surfacing', 'flippers']
    test = [1,1]
    #lb = tool.classify(tree, test=test)
    lb = tool.classify_label(tree, feature_label, test)

    print(str('the test {} is {}\n').format(test, lb))

    data_load = load_data.utility()
    lense = data_load.load('./lenses.txt')
    feature_label = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = tool.create_tree(lense, feature_label)
    print(str('the tree is {}\n').format(tree))

