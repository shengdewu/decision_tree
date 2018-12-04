import matplotlib.pyplot as plt

class utility(object):

    def load(self, path):
        with open(path) as d_file:
            lense = [line.strip().split('\t') for line in d_file.readlines()]
        return lense