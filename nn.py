import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


def default_initializer(std=0.05):
    # return tf.contrib.layers.xavier_initializer()
    # return tf.random_uniform_initializer()
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1] + list(map(int, x.get_shape()[1:]))


def flatten_sum(logps, keepdims=False):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1], keepdims=keepdims)
    elif len(logps.get_shape()) == 3:
        return tf.reduce_sum(logps, [1, 2], keepdims=keepdims)
    else:
        raise Exception()


# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init
@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        assign_w = w.assign(initial_value)
        with tf.control_dependencies([assign_w]):
            return w
    return w


# Activation normalization
# Convenience function that does centering+scaling
@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if logdet is not None:
            if not reverse:
                x = actnorm_center(name + "_center", x, reverse)
                x, logdet = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor, batch_variance, reverse)
            else:
                x, logdet = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor, batch_variance, reverse)
                x = actnorm_center(name + "_center", x, reverse)
            return x, logdet
        else:
            if not reverse:
                x = actnorm_center(name + "_center", x, reverse)
                x = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor, batch_variance, reverse)
            else:
                x = actnorm_scale(name + "_scale", x, scale, logdet, logscale_factor, batch_variance, reverse)
                x = actnorm_center(name + "_center", x, reverse)
            return x


# Activation normalization
@add_arg_scope
def actnorm_center(name, x, reverse=False):
    x_shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(x_shape) == 2 or len(x_shape) == 3
        if len(x_shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi("b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(x_shape) == 3:
            x_mean = tf.reduce_mean(x, [0, 1], keepdims=True)
            b = get_variable_ddi("b", (1, 1, int_shape(x)[2]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x


# Activation normalization
@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False,
                  trainable=True):
    x_shape = int_shape(x)
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(x_shape) == 2 or len(x_shape) == 3

        if len(x_shape) == 2:
            x_var = tf.reduce_mean(x ** 2, [0], keepdims=True)
            logdet_factor = 1
            _shape = [1, x_shape[1]]
        elif len(x_shape) == 3:
            x_var = tf.reduce_mean(x ** 2, [0, 1], keepdims=True)
            logdet_factor = int(x_shape[1])
            _shape = [1, 1, x_shape[2]]

        if batch_variance:
            x_var = tf.reduce_mean(x ** 2, keepdims=True)

        logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
            scale / (tf.sqrt(x_var) + 1e-6)) / logscale_factor) * logscale_factor

        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x / tf.exp(logs)

        if logdet is not None:
            dlogdet = tf.reduce_sum(tf.log(tf.abs(tf.exp(logs)))) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x


def add_edge_padding(x, filter_size):
    assert filter_size % 2 == 1
    if filter_size == 1:
        return x
    a = (filter_size - 1) // 2  # width padding size

    pad = tf.pad(
        tf.zeros_like(x[:, :, :1]) - 1,
        [[0, 0],
         [a, a],
         [0, 0]]
    ) + 1
    x = tf.pad(x, [[0, 0], [a, a], [0, 0]])
    x = tf.concat([x, pad], axis=2)
    return x


@add_arg_scope
def conv1d(name, x, n_out, filter_size=3, stride=1, padding="SAME", do_weightnorm=False, do_actnorm=True,
           edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and padding == "SAME":
            x = add_edge_padding(x, filter_size)
            padding = 'VALID'

        n_in = int(x.get_shape()[2])

        filter_shape = [filter_size, n_in, n_out]
        w = tf.get_variable("W", filter_shape, tf.float32, initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1])

        x = tf.nn.conv1d(x, w, stride, padding, data_format='NWC')

        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, n_out], initializer=tf.zeros_initializer())

    return x


@add_arg_scope
def conv1d_zeros(name, x, n_out, filter_size=1, stride=1, padding="SAME", logscale_factor=3, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and padding == "SAME":
            x = add_edge_padding(x, filter_size)
            padding = "VALID"

        n_in = int(x.get_shape()[2])
        w = tf.get_variable("W", [filter_size, n_in, n_out], tf.float32, initializer=tf.zeros_initializer())

        logs = tf.get_variable("logs", [1, 1, n_out], initializer=tf.zeros_initializer()) * logscale_factor

        x = tf.nn.conv1d(x, w, stride, padding, data_format='NWC')
        x += tf.get_variable("b", [1, 1, n_out], initializer=tf.zeros_initializer())
        x *= tf.exp(logs)
    return x


class GaussianDiag:
    def __init__(self, mean, logsd):
        self.mean = mean
        self.logsd = logsd

    def sample(self, eps):
        return self.mean + tf.exp(self.logsd) * eps
        # return tf.random_normal(tf.shape(eps), self.mean, tf.exp(self.logsd))

    def logp(self, x):
        return -0.5 * (np.log(2 * np.pi) + 2. * self.logsd + (x - self.mean) ** 2 / tf.exp(2. * self.logsd))

    def get_eps(self, x):
        return (x - self.mean) / tf.exp(self.logsd)


class GaussianPrior:
    def __init__(self, name, data_shape):
        with tf.variable_scope(name):
            n_z = data_shape[1]
            h = tf.zeros([1, data_shape[0], 2 * n_z])
            h = conv1d_zeros('p', h, 2 * n_z)

            mean = h[:, :, :n_z]
            logsd = h[:, :, n_z:]

            self.pz = GaussianDiag(mean, logsd)

    def mean(self):
        return self.pz.mean

    def logsd(self):
        return self.logsd

    def logp(self, z1):
        objective = self.pz.logp(z1)
        return objective

    def sample(self, eps):
        return self.pz.sample(eps)

    def eps(self, z1):
        return self.pz.get_eps(z1)


def f(name, h, hidden_size, n_out=None):
    n_out = n_out or int(h.get_shape()[2])
    with tf.variable_scope(name):
        h = tf.nn.relu(conv1d("l_1", h, hidden_size, filter_size=1))
        h = tf.nn.relu(conv1d("l_2", h, hidden_size, filter_size=1))
        h = conv1d_zeros("l_last", h, n_out)
    return h


def f_dense(name, h, hidden_size):
    n_w = int(h.get_shape()[1])
    n_z = int(h.get_shape()[2])
    with tf.variable_scope(name):
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, units=hidden_size, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=hidden_size, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=n_w * n_z)
        h = tf.reshape(h, [-1, n_w, n_z])
    return h


# Invertible 1x1 conv
@add_arg_scope
def inv1x1conv(name, z, logdet, reverse=False):
    with tf.variable_scope(name):

        shape = int_shape(z)
        w_shape = [shape[2], shape[2]]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)
        dlogdet = tf.cast(tf.log(tf.abs(tf.matrix_determinant(tf.cast(w, 'float64')))), 'float32') * shape[1]

        if not reverse:
            _w = tf.reshape(w, [1] + w_shape)
            z = tf.nn.conv1d(z, _w, 1, 'SAME', data_format='NWC')
            logdet += dlogdet

        else:
            w_inv = tf.matrix_inverse(w)
            _w = tf.reshape(w_inv, [1] + w_shape)
            z = tf.nn.conv1d(z, _w, 1, 'SAME', data_format='NWC')
            logdet -= dlogdet

        return z, logdet


def affine_coupling(name, z, logdet, hidden_size, kind=0, reverse=False):
    with tf.variable_scope(name):
        shape = int_shape(z)
        n_w = shape[1]
        n_z = shape[2]

        assert n_z % 2 == 0

        if kind == 0:
            z1 = z[:, :, :n_z // 2]
            z2 = z[:, :, n_z // 2:]
        elif kind == 1:
            z2 = z[:, :, :n_z // 2]
            z1 = z[:, :, n_z // 2:]

        h = f("f2", z1, hidden_size, n_z)
        shift = h[:, :, 0::2]
        scale = tf.add(tf.nn.sigmoid(h[:, :, 1::2] + 2.), 1e-6, "scale")

        if not reverse:
            z2 += shift
            z2 = z2 * scale

            dlogdet = tf.reduce_sum(tf.log(tf.abs(scale)), axis=[1, 2], keepdims=True)
            logdet += dlogdet
            z = tf.concat([z1, z2], 2)
        else:
            z2 = z2 / scale
            z2 -= shift

            dlogdet = tf.reduce_sum(tf.log(tf.abs(scale)), axis=[1, 2], keepdims=True)
            logdet -= dlogdet
            z = tf.concat([z1, z2], 2)

        return z, logdet


@add_arg_scope
def flow_step(name, z, logdet, hidden_size, reverse):
    with tf.variable_scope(name):
        if not reverse:
            z, logdet = actnorm("1", z, logdet=logdet, reverse=False)
            z, logdet = inv1x1conv("2", z, logdet, reverse=False)
            z, logdet = affine_coupling("3", z, logdet, hidden_size, kind=0, reverse=False)
            z, logdet = actnorm("4", z, logdet=logdet, reverse=False)
            z, logdet = affine_coupling("5", z, logdet, hidden_size, kind=1, reverse=False)
        else:
            raise Exception("Unimplemented")
    return z, logdet


@add_arg_scope
def normalizing_flow(name, z, logdet, hidden_size, depth, reverse=False):
    with tf.variable_scope(name):
        for i in range(depth):
            z, logdet = flow_step(str(i), z, logdet, hidden_size, reverse)
    return z, logdet
