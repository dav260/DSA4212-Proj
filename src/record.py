import time
import numpy as np
from .objectives import objective

class Recorder:
    def __init__(self, name, D, d, budget):
        self.name, self.D, self.d, self.budget = name, D, d, budget
        self.rows = []
        self.start = time.perf_counter()
    def evaluate(self, x):
        if len(self.rows) >= self.budget:
            raise RuntimeError("Objective evaluation budget exceeded")
        x = np.asarray(x, dtype=float).copy()
        y = objective(x, self.name, self.D, self.d)
        best = min(y, self.rows[-1]['best_so_far']) if self.rows else y
        self.rows.append(dict(evaluation=len(self.rows)+1, x=x.tolist(), value=y,
                              best_so_far=best, elapsed_seconds=time.perf_counter()-self.start))
        return y
