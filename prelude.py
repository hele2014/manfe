import os
import pathlib

from global_settings import *


def get_bits(x):
    return np.where(x < 0, 0, 1)


def check_wrong_bits(bits, bits_estimated):
    return len(np.argwhere(bits != bits_estimated))


def mkdir(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def mkfile(file_path):
    mkdir(file_path)
    filename = pathlib.Path(file_path)
    filename.touch(exist_ok=True)


def concatenate(total, part):
    return part if total is None else np.concatenate((total, part))


def complex_channel(m=NUM_ANT, n=NUM_ANT):
    real = np.random.randn(m, n)
    imag = np.random.randn(m, n)
    h = np.row_stack(
        (
            np.column_stack((real, -imag)),
            np.column_stack((imag, real)),
        )
    )
    return h


def make_channel_batch():
    h_batch = None
    for _ in range(PACKETS_PER_BATCH):
        h = complex_channel().reshape([1, 2 * NUM_ANT, 2 * NUM_ANT])
        for _ in range(TIME_SLOTS_PER_PACKET):
            h_batch = concatenate(h_batch, h)
    return h_batch


def signal_batch(batch_size=TIMES_SLOTS_PER_BATCH):
    s_batch = None
    random_indexes = np.random.uniform(low=0, high=QPSK_CANDIDATE_SIZE, size=batch_size)
    for t in range(batch_size):
        i = int(random_indexes[t])
        s = QPSK_CANDIDATES[:, i:i + 1].reshape([1, 2 * NUM_ANT, 1])
        s_batch = concatenate(s_batch, s)
    return s_batch


def random_distance(n, length):
    x = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    y = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    return np.sqrt(x ** 2 + y ** 2)


def zf_batch(y, h):
    h_t = np.transpose(h, axes=[0, 2, 1])
    f = np.linalg.inv(h_t @ h) @ h_t
    z = f @ y
    return np.where(z < 0, -1, 1) / np.sqrt(2)


def lmmse_batch(y, h):
    assert len(h.shape) == 3
    batch_size, m, n = h.shape
    eye = np.concatenate([np.eye(n).reshape([1, n, n]) * batch_size], axis=0)
    ht = np.transpose(h, axes=[0, 2, 1])
    z = np.linalg.inv(ht @ h + eye) @ ht @ y
    return np.where(z < 0, -1, 1) / np.sqrt(2)


def maximum_likelihood_detect_bits(y, h):
    assert len(h.shape) == 3
    batch_size, m, n = h.shape
    s_mld = np.zeros([batch_size, n, 1])

    if True:
        dst = np.sum(np.square(y - h @ QPSK_CANDIDATES), axis=1)
    else:
        dst = None
        for j in range(QPSK_CANDIDATE_SIZE):
            s_cand = QPSK_CANDIDATES[:, j:j + 1].reshape([1, 2 * NUM_ANT, 1])
            dj = np.sum(np.square(y - h @ s_cand), axis=(1, 2)).reshape([-1, 1])

            if dst is None:
                dst = dj
            else:
                dst = np.concatenate((dst, dj), axis=1)

    min_indexes = dst.argmin(1)
    for i, t in enumerate(min_indexes):
        s_mld[i:i + 1, :, :] = QPSK_CANDIDATES[:, t].reshape([1, 2 * NUM_ANT, 1])

    return get_bits(s_mld)


