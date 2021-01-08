import tensorflow as tf

import nn
from abstract_model import AbstractModel
from message_passing import *


def squeeze(x):
    r = x[:, :NUM_ANT, :]
    i = x[:, NUM_ANT:, :]
    return tf.concat([r, i], axis=2)


def unsqueeze(x):
    r = x[:, :, 0:1]
    i = x[:, :, 1:2]
    return tf.concat([r, i], axis=1)


class MANFE(AbstractModel):
    """
    Maximum a normalizing flow estimate
    """

    def __init__(self, alpha, hidden_size, depth):
        self.hidden_size = hidden_size
        self.depth = depth
        self.__masks = {}

        name = "MANFE_qpsk_ant{}_hidden{}_depth{}_alpha{}".format(NUM_ANT, hidden_size, depth, alpha)
        super(MANFE, self).__init__(name)

    def _build_graph(self):
        self.optim = tf.train.AdamOptimizer()
        with tf.variable_scope("input"):
            self.w = tf.placeholder(tf.float32, [None, 2 * NUM_ANT, 1], name='w')
            self.eps_sample = tf.placeholder(tf.float32, [None, 2 * NUM_ANT, 1], name='eps_sample')
        self._logprob(self.w)

    def _logprob(self, w):
        with tf.variable_scope("inference_model"):
            logdet = tf.zeros([tf.shape(w)[0], 1, 1], dtype='float32')
            prior = nn.GaussianPrior("gaussian_prior", [2 * NUM_ANT, 1])

            z = squeeze(w)
            z, logdet = nn.normalizing_flow("gflow", z, logdet, self.hidden_size, self.depth, reverse=False)
            self.z = unsqueeze(z)
            self.logpz = prior.logp(self.z)
            self.logpw = nn.flatten_sum(self.logpz, keepdims=True) + logdet

            self.loss = tf.reduce_mean(-self.logpw)
            tf.summary.scalar("loss", self.loss)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inference_model")
            self.train_op = self.optim.minimize(self.loss, var_list=var_list)

    def _train_func(self, batch_idx, batch_data):
        w = batch_data
        _, loss = self._sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.w: w.reshape([-1, 2 * NUM_ANT, 1]),
            }
        )
        print("Training {}, batch {}, loss={:e}".format(self.name, batch_idx + 1, loss), end="\r")

    def _test_func(self, batch_idx, batch_data):
        w = batch_data
        loss = self._sess.run(
            self.loss,
            feed_dict={
                self.w: w.reshape([-1, 2 * NUM_ANT, 1]),
            }
        )
        print("Testing, batch {}, nll={:e}".format(batch_idx, loss), end="\r")
        return loss

    def logprob(self, w):
        logpw = self._sess.run(
            self.logpw,
            feed_dict={
                self.w: w.reshape([-1, 2 * NUM_ANT, 1]),
            }
        )
        return logpw

    def detect_bits(self, y, h):
        assert len(h.shape) == 3
        batch_size, m, n = h.shape
        s_est = np.zeros([batch_size, n, 1])

        likelihoods = np.zeros([batch_size, QPSK_CANDIDATE_SIZE])
        w_cands = y - h @ np.reshape(QPSK_CANDIDATES, [1, 2 * NUM_ANT, QPSK_CANDIDATE_SIZE])
        for t in range(QPSK_CANDIDATE_SIZE):
            logp = self.logprob(w_cands[:, :, t:t + 1])
            likelihoods[:, t:t + 1] = logp.reshape([batch_size, 1])

        indexes = np.argmax(likelihoods, axis=1)
        for i, t in enumerate(indexes):
            s_est[i:i + 1, :, :] = QPSK_CANDIDATES[:, t].reshape([1, 2 * NUM_ANT, 1])

        return get_bits(s_est)

    def __get_qpsk_error_mask(self, no_err):
        if no_err == 0:
            identity_mask = np.ones([1, 2 * NUM_ANT, 1])
            self.__masks[no_err].append(identity_mask)
        elif no_err > 0:
            iden = self.__masks[0][0]
            all_pos = [x for x in itertools.combinations([y for y in range(NUM_ANT)], no_err)]
            all_kinds = [x for x in itertools.product([0, 1, 2], repeat=no_err)]
            for pos in all_pos:
                for kind in all_kinds:
                    mask = np.copy(iden)
                    for i, j in enumerate(kind):
                        idx = pos[i]
                        if j == 0:
                            mask[:, idx, :] *= -1
                        elif j == 1:
                            mask[:, idx+NUM_ANT, :] *= -1
                        else:
                            mask[:, idx, :] *= -1
                            mask[:, idx + NUM_ANT, :] *= -1
                    self.__masks[no_err].append(mask)
        else:
            raise ValueError(no_err)

    def detect_bits_with_initial_guess(self, y, h, initial_guess, max_error_symbols, use_mld=False):
        assert max_error_symbols <= NUM_ANT
        assert len(h.shape) == 3
        batch_size, m, n = h.shape
        s_est = np.zeros([batch_size, n, 1])

        # perform map estimate
        s_cand = []
        l_cand = []
        for n_err in range(0, max_error_symbols + 1):
            if n_err not in self.__masks:
                self.__masks[n_err] = []
                self.__get_qpsk_error_mask(n_err)

            for mask in self.__masks[n_err]:
                s_possible = initial_guess * mask
                if not use_mld:
                    l_possible = self.logprob(y - h @ s_possible)
                else:
                    dst = (y - h @ s_possible)**2
                    l_possible = np.sum(dst, axis=1, keepdims=True)
                s_cand.append(s_possible)
                l_cand.append(l_possible)

        s_cand = np.concatenate(s_cand, axis=2)
        l_cand = np.concatenate(l_cand, axis=1)

        if not use_mld:
            indexes = np.argmax(l_cand, axis=1)
        else:
            indexes = np.argmin(l_cand, axis=1)

        for i, t in enumerate(indexes.flatten()):
            s_est[i:i + 1, :, :] = s_cand[i:i + 1, :, t:t + 1].reshape([1, 2 * NUM_ANT, 1])

        return get_bits(s_est)
