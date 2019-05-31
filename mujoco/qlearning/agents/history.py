import numpy as np


class History:
    def __init__(self, data_format, batch_size, screen_dims):
        self.data_format = data_format
        self.history = np.zeros(screen_dims, dtype=np.float32)

    def add(self, screen):
        #  self.history[:, :, 0:9] = self.history[:, :, 3:12]
        self.history = screen

    def reset(self):
        self.history *= 0

    def get(self):
        #  if self.data_format == 'NHWC' and len(self.history.shape) == 3:
            #  return self.history
        #  else:
            #  return np.transpose(self.history, (2, 0, 1))
        return self.history
