import torch
import torch.nn as nn
import numpy as np


class WeightLoader(nn.Module):
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None

    def load(self, model, weights_path):
        self.start = 0
        fp = open(weights_path, 'rb')
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size()
        self.dfs(model)

        assert size == self.start