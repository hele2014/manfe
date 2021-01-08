import tensorflow as tf

from prelude import *


class DetNet:
    def __init__(self, num_ant, hidden_size, iteration, alpha, snr):
        self.name = "DetNet_ant{}_hidden{}_iter{}_alpha{}_snr{}".format(NUM_ANT, hidden_size, iteration, alpha, snr)
        self.num_ant = num_ant
        self.hidden_size = hidden_size
        self.iteration = iteration

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self._build_graph(), config=tf.ConfigProto(gpu_options=gpu_options))

    def _build_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.adam = tf.train.AdamOptimizer()

            self.y = tf.placeholder(tf.float32, [None, 2 * self.num_ant, 1], name='y')
            self.h = tf.placeholder(tf.float32, [None, 2 * self.num_ant, 2 * self.num_ant], name='h')
            self.soh_label = tf.placeholder(tf.float32, [None, self.num_ant, 4], name='soh_label')

            self.train_op = self.__build_detect_model()

            self.merge_all = tf.summary.merge_all()
            self.init_global_var = tf.global_variables_initializer()

        return self.graph

    def __build_detect_model(self):
        with tf.variable_scope("det_net"):
            self.loss = tf.constant(0.0, name="loss")
            s_est = tf.zeros_like(self.y, name="s_estimated")
            v = tf.zeros_like(s_est, name="v")

            ht = tf.transpose(self.h, perm=[0, 2, 1])
            hth = ht @ self.h
            hty = ht @ self.y

            for i in range(self.iteration):
                with tf.variable_scope("layer_{}".format(i)):
                    with tf.variable_scope("linear_estimation"):
                        alpha1 = tf.get_variable("alpha1", initializer=tf.constant(0.1))
                        alpha2 = tf.get_variable("alpha2", initializer=tf.constant(0.1))
                        r = s_est - alpha1 * hty + alpha2 * hth @ s_est

                    with tf.variable_scope("nonlinear_estimation"):
                        q = tf.layers.flatten(tf.concat([r, v], axis=1))
                        q = tf.layers.dense(q, units=self.hidden_size, activation=tf.nn.relu)

                    with tf.variable_scope("update_s"):
                        s_oh_new = tf.reshape(tf.layers.dense(q, units=self.num_ant * 4), [-1, self.num_ant, 4])

                        with tf.control_dependencies([
                            tf.check_numerics(s_oh_new, "chk_s_oh_new_{}".format(i)),
                        ]):
                            if i == 0:
                                s_oh = s_oh_new
                            else:
                                s_oh = 0.9 * s_oh + 0.1 * s_oh_new

                        s_oh = tf.clip_by_value(s_oh, 0, 1)

                        real = -1 * s_oh[:, :, 0] - 1 * s_oh[:, :, 1] + 1 * s_oh[:, :, 2] + 1 * s_oh[:, :, 3]
                        imag = -1 * s_oh[:, :, 0] + 1 * s_oh[:, :, 1] - 1 * s_oh[:, :, 2] + 1 * s_oh[:, :, 3]
                        s_est = tf.concat([real, imag], axis=1) / np.sqrt(2)
                        s_est = tf.reshape(s_est, [-1, 2 * self.num_ant, 1])

                    with tf.variable_scope("update_v"):
                        v_next = tf.reshape(tf.layers.dense(q, units=2 * self.num_ant), [-1, 2 * self.num_ant, 1])

                        with tf.control_dependencies([
                            tf.check_numerics(v_next, "chk_v_next_{}".format(i)),
                        ]):
                            v = 0.9 * v + 0.1 * v_next

                    with tf.variable_scope("add_loss"):
                        sub_loss = tf.reduce_mean(
                            tf.reduce_mean(tf.square(s_oh - self.soh_label), axis=[1, 2]),
                            name="sub_loss"
                        )
                        tf.summary.scalar("sub_loss_{}".format(i), sub_loss)
                        self.loss += tf.log(float(i + 2)) * sub_loss

                tf.summary.scalar("loss", self.loss)

            self.final_est = s_est

        return self.adam.minimize(self.loss)

    def train(self, train_io, test_io, max_flip, max_epoch):
        print("Initializing model {} ...".format(self.name))
        self.sess.run(self.init_global_var)

        # self.load()

        tf_writer = self._create_writer()

        flip_count = 0
        best_loss = None
        epoch = 0
        while epoch < max_epoch:
            batch_idx = 0
            for y, h, s, soh, w in train_io.fetch():
                _, loss, merge_all = self.sess.run(
                    [self.train_op, self.loss, self.merge_all],
                    feed_dict={
                        self.y: y.reshape([-1, 2 * self.num_ant, 1]),
                        self.h: h.reshape([-1, 2 * self.num_ant, 2 * self.num_ant]),
                        self.soh_label: soh.reshape([-1, self.num_ant, 4]),
                    }
                )

                tf_writer.add_summary(merge_all)

                print("Training {}, epoch {}, batch {}, loss={:e}".format(
                    self.name, epoch + 1, batch_idx + 1, loss), end='\r')

                if "{}".format(loss) == "inf" or "{}".format(loss) == "-inf" or "{}".format(loss) == "nan":
                    raise Exception("Invalid loss detected")

                batch_idx += 1

            print()

            new_loss = self._test(test_io)
            if best_loss is None or new_loss < best_loss:
                best_loss = new_loss
                print("{} validated, new_loss={} best_loss={}".format(self.name, new_loss, best_loss))
                self._save()
            else:
                print("{} validated, new_loss={} best_loss={}".format(self.name, new_loss, best_loss))
                flip_count += 1
                if flip_count >= max_flip:
                    break

            epoch += 1
        print("Model '{}' train over".format(self.name))

    def _test(self, test_io):
        total_loss = 0.0
        count = 0
        for y, h, s, soh, w in test_io.fetch():
            loss = self.sess.run(
                self.loss,
                feed_dict={
                    self.y: y.reshape([-1, 2 * self.num_ant, 1]),
                    self.h: h.reshape([-1, 2 * self.num_ant, 2 * self.num_ant]),
                    self.soh_label: soh.reshape([-1, self.num_ant, 4]),
                }
            )

            print("Testing, batch {}".format(count + 1), end="\r")

            total_loss += loss
            count += 1

        print()

        avg_loss = total_loss / count
        return avg_loss

    def _create_writer(self):
        path = "SavedModel/Board/{}/".format(self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        return tf.summary.FileWriter(
            logdir=path,
            graph=self.graph,
            session=self.sess
        )

    def load(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "SavedModel/{}/".format(self.name)
            saver.restore(self.sess, path)
            print("Model \"{}\" loaded".format(self.name))

    def close(self):
        self.sess.close()

    def _save(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "SavedModel/{}/".format(self.name)
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            prefix = saver.save(self.sess, path)
            print("Model saved at \"{}\"".format(prefix))

    def detect_bits(self, y, h):
        s_detected = self.sess.run(
            self.final_est,
            feed_dict={
                self.y: y,
                self.h: h,
            }
        )

        return get_bits(s_detected)
