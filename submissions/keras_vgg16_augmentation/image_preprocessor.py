import numpy as np
from skimage.transform import resize


def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    h, w = x.shape[:2]
    min_shape = min(h, w)
    x = x[h / 2 - min_shape / 2:h / 2 + min_shape / 2,
          w / 2 - min_shape / 2:w / 2 + min_shape / 2]
    x = resize(x, (64, 64), preserve_range=True)
    # random crop of size 64x64 with padding of 8
    x = np.pad(x, ((8,8),(8,8),(0,0)), mode='constant')
    row, col = np.random.randint(0, 16, size=2)
    x = x[row: row + 64, col: col + 64, :]
    x = x / 255.
    return x


def transform_test(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    h, w = x.shape[:2]
    min_shape = min(h, w)
    x = x[h / 2 - min_shape / 2:h / 2 + min_shape / 2,
          w / 2 - min_shape / 2:w / 2 + min_shape / 2]
    x = resize(x, (64, 64), preserve_range=True)
    x = x / 255.
    return x
