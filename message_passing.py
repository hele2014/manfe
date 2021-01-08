import mpmath

from prelude import *


def guass_fz(v, o, y):
    z = o + v * (y - o) / (1 + v)
    z_var = 1 * v / (1 + v)
    return z, z_var


def fx_mmse(s, r):
    x = np.zeros_like(s)
    x_var = np.zeros_like(r)

    px = 0.5
    for i in range(2 * NUM_ANT):
        sum_n1 = 0
        sum_n2 = 0
        sum_norm = 0

        s_i = float(s[i, 0])
        r_i = float(r[i, 0])

        for x_cand in [-1 / mpmath.sqrt(2), 1 / mpmath.sqrt(2)]:
            tmp = mpmath.exp(-0.5 * (x_cand - r_i) ** 2 / s_i)
            pr_xcand = tmp / mpmath.sqrt(2 * mpmath.pi * s_i)
            norm = px * pr_xcand
            n1 = x_cand * norm
            n2 = 0.5 * norm

            sum_norm += norm
            sum_n1 += n1
            sum_n2 += n2

        x_i = float(sum_n1 / sum_norm)
        x_var_i = float(sum_n2 / sum_norm - x_i ** 2)
        x[i, 0] = x_i
        x_var[i, 0] = x_var_i

    return x, x_var


def mmse_denoiser(s, r):
    px = 0.5
    sum_n1 = 0
    sum_n2 = 0
    sum_norm = 0

    try:
        for x_cand in [-1 / np.sqrt(2), 1 / np.sqrt(2)]:
            tmp = np.exp(-0.5 * (x_cand - r) ** 2 / s)
            pr_xcand = tmp / np.sqrt(2 * np.pi * s)
            norm = px * pr_xcand
            n1 = x_cand * norm
            n2 = 0.5 * norm

            sum_norm += norm
            sum_n1 += n1
            sum_n2 += n2

        x = sum_n1 / sum_norm
        x_var = sum_n2 / sum_norm - x ** 2
        return x, x_var
    except:
        return None, None


def amp_detect_unstable(y, H, maxloop):
    """
    exp overflow happens in high SNR
    """
    np.seterr(all='raise')
    m, n = H.shape

    Hasq = np.square(H)
    Hasq_H = Hasq.T
    H_H = H.T

    q = np.zeros([m, 1])
    x = np.zeros([n, 1])
    x_var = np.zeros([2 * NUM_ANT, 1]) + 1 / np.sqrt(2)

    i = 0
    while i < maxloop:
        # Output node
        v = Hasq @ x_var
        o = H @ x - v * q
        z, z_var = guass_fz(v, o, y)

        q = (z - o) / v
        u = (v - z_var) / np.square(v)

        # Input node
        s = 1 / (Hasq_H @ u)
        r = x + s * (H_H @ q)
        x_next, x_var_next = mmse_denoiser(s, r)
        i += 1

        if x_next is None:
            break
        else:
            x = x_next

    np.seterr(all='warn')
    return np.where(x < 0, -1, 1) / np.sqrt(2), i


def amp_detect_stable(y, H, loop):
    """
    Numerical stable generalized approximate message passing with damping
    """
    np.seterr(all='raise')
    m, n = H.shape

    Hasq = np.square(H)
    Hasq_H = Hasq.T
    H_H = H.T

    x = np.zeros([n, 1])
    hat_x = np.zeros([n, 1])
    x_var = np.zeros([2 * NUM_ANT, 1]) + 0.5
    q = np.zeros([m, 1])
    v = np.zeros([m, 1])
    u = np.zeros([m, 1])

    best_x = x
    best_x_var = x_var

    k = 0.02

    i = 0
    while i < loop:
        # Factor node
        v = k * Hasq @ x_var + (1 - k) * v
        mu = 1 / m * v.T @ np.ones([m, 1])
        o = H @ x - 1 / mu * v * q
        z, z_var = guass_fz(v, o, y)

        q = k * mu * (z - o) / v + (1 - k) * q
        u = k * mu * (v - z_var) / np.square(v) + (1 - k) * u

        # Variable node
        hat_x = k * x + (1 - k) * hat_x
        s = 1 / (Hasq_H @ u)
        r = hat_x + s * (H_H @ q)
        x_next, x_var_next = mmse_denoiser(s, r)
        if x_next is not None and x_var_next is not None:
            x = x_next
            x_var = mu * x_var_next
            if np.sum((y - H @ x) ** 2) < np.sum((y - H @ best_x) ** 2):
                best_x = x
                best_x_var = x_var
        else:
            x = best_x
            x_var = best_x_var

        i += 1
    np.seterr(all='warn')
    return np.where(x < 0, -1, 1) / np.sqrt(2), i


def amp_batch(y, h, loop):
    assert len(h.shape) == 3
    batch_size, m, n = h.shape

    s_est = []
    t = 0
    for i in range(batch_size):
        y_ = y[i, :, :]
        h_ = h[i, :, :]
        x_, t_ = amp_detect_unstable(y_, h_, loop)
        s_est.append(x_.reshape([1, n, 1]))
        t += t_
    s_est = np.concatenate(s_est, axis=0)
    return np.where(s_est < 0, -1, 1) / np.sqrt(2), t / batch_size
