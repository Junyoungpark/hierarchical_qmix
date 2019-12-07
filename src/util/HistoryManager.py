import dgl
from collections import deque


class HistoryManager:

    def __init__(self, n_hist_steps, init_graph):
        self.n_hist_steps = n_hist_steps
        self.hist = deque(maxlen=n_hist_steps)
        self.reset(init_graph)

    def append(self, graph):
        self.hist.append(graph)

    def get_hist(self):
        return dgl.batch([g for g in self.hist])

    def reset(self, init_graph):
        self.hist.clear()
        for _ in range(self.n_hist_steps):
            self.hist.append(init_graph)
