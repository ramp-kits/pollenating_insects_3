from skimage.transform import resize


def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    h, w = x.shape[:2]
    min_shape = min(h, w)
    x = x[h / 2 - min_shape / 2:h / 2 + min_shape / 2,
          w / 2 - min_shape / 2:w / 2 + min_shape / 2]

    x = resize(x, (64, 64), preserve_range=True)
    x /= 255.
    x = x.transpose((2, 0, 1))
    return x

def transform_test(x):
    return transform(x)
