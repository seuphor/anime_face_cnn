import numpy as np

class batch_selection():
    def __init__(self):
        self.current_index = 0
    
    def next_batch(self, x, y, batch_size):
        assert(x.shape[0] == y.shape[0])
        start = self.current_index
        
        if start == 0:
            perm = np.arange(x.shape[0])
            np.random.shuffle(perm)
            self._x = x[perm]
            self._y = y[perm]
        
        if (start + batch_size) >= x.shape[0]:
            self.current_index = 0
            return self._x[start:x.shape[0]], self._y[start:x.shape[0]]
        
        else:
            self.current_index += batch_size
            end = self.current_index
            return self._x[start:end], self._y[start:end]

    def next_batch_single(self, x, batch_size):
        #assert(x.shape[0] == y.shape[0])
        start = self.current_index
        
        if start == 0:
            perm = np.arange(x.shape[0])
            np.random.shuffle(perm)
            self._x = x[perm]
            #self._y = y[perm]
        
        if (start + batch_size) >= x.shape[0]:
            self.current_index = 0
            return self._x[start:x.shape[0]]
        
        else:
            self.current_index += batch_size
            end = self.current_index
            return self._x[start:end]
