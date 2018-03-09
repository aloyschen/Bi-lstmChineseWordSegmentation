import numpy as np
ary = np.sin(np.arange(10).reshape([2, 5]))
print(ary[:, :, np.newaxis])