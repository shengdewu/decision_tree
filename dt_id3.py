import numpy as np
from math import log2
import operator

class utility(object):

    def calc_entropy(self, data_set):
        '''
        :param data_set: [x1,x2,x3....label] type list
        :return:
        '''

        data_set = np.array(data_set)
        label_set = data_set[: ,-1]
        total = len(label_set)
        label_cnt = {}
        for label in label_set:
            num = label_cnt.get(label, -1)
            if -1 == num:
                num = 0
            label_cnt[label] = num + 1

        ent = 0
        for key, val in label_cnt.items():
            p = val / total
            info = -(p * log2(p))
            ent = ent + info

        return ent

    def split_data(self, data_set, ax, value):
        ret_cate = []
        for data in data_set:
            if data[ax] == value:
                fh = data[:ax]
                fh.extend(data[ax+1:])
                ret_cate.append(fh)

        return ret_cate

    def calc_best_ft(self, data_set):
        ent = self.calc_entropy(data_set)
        ft_num = len(data_set[0]) - 1
        data_num = len(data_set)
        best_ent = 0
        best_num = -1
        for num in range(ft_num):
            ft_list = [ft[num] for ft in data_set]
            ft_unq = set(ft_list)
            gain_ent = 0
            for ft in ft_unq:
                sub_data = self.split_data(data_set, num, ft)
                sub_ent = self.calc_entropy(sub_data)
                sub_num = len(sub_data)
                gain_ent = gain_ent + (sub_ent * sub_num/data_num)
            gain_ent = ent - gain_ent

            if best_ent < gain_ent:
                best_ent = gain_ent
                best_num = num
        return best_num

    def majorcnt(self, data_set):
        ft_num = {}
        for data in data_set:
            num = ft_num.get(data, -1)
            if -1 == num:
                num = 0
            ft_num[data] = num + 1

        ft_num = sorted(ft_num, key=operator.itemgetter(1), reverse=True)

        return ft_num[0][0]

    def create_tree(self, data_set, ft_label=None):
        '''
        :param data_set:
        :param ft_label: 特征向量标签
        :return:
        '''
        classList = [data[-1] for data in data_set]
        #t同一类
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        #整个数据集只有标签时
        if len(data_set[0]) == 1:
            return self.majorcnt(classList)

        b_ft_ax = self.calc_best_ft(data_set)
        ft_lb = ft_label[b_ft_ax]
        del(ft_label[b_ft_ax])
        b_ft = [ft[b_ft_ax] for ft in data_set]
        b_ft = set(b_ft)
        m_tree = {ft_lb:{}}
        for bt in b_ft:
            c_ft = ft_label.copy()
            sd = self.split_data(data_set, b_ft_ax, bt)
            m_tree[ft_lb][bt] = self.create_tree(sd, c_ft)

        return m_tree

    def classify(self, tree, test=None):
        '''
        :param tree:
        :param test:
        :return:
        算法每次运算只用到一个特征，没有考虑到全部
        比如本例 每下降一层，就会少一个特征，相当于其中一个子节点没有起到分类的作用
        '''
        f_lb = list(tree.keys())[0]
        s_dict = tree[f_lb]

        lb = ''
        for key,value in s_dict.items():
            if key == test[0]:
                if type(value).__name__ != 'dict':
                    lb = value
                else:
                    lb = self.classify(value, test[1:])

        return lb

    def classify_label(self, inTree, ft_label, testVec):
        firstKey = list(inTree.keys())[0]
        featIndex = ft_label.index(firstKey)
        key = testVec[featIndex]
        subTree = inTree[firstKey]
        featLabel = subTree[key]
        if type(featLabel).__name__ == 'dict':
            classLabel = self.classify_label(featLabel, ft_label, testVec)
        else:
            classLabel = featLabel

        return classLabel

    def get_tree_info(self, tree, fa_lay):
        node_info = []
        if type(tree) != dict:
            return
        lay_num = fa_lay + 1
        for key, value in tree.items():
            if type(value) == dict:
                node_info.extend(self.get_tree_info(value, lay_num))
            else:
                node_info.append((value, lay_num))

        return node_info





