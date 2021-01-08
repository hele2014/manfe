import abc

import tensorflow as tf

from prelude import *


class AbstractModel(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._build_graph()
            self.__summary_op = tf.summary.merge_all()
            self.__init_op = tf.global_variables_initializer()

        self._sess = tf.Session(graph=self._graph, config=tf.ConfigProto(gpu_options=gpu_options))

    @abc.abstractmethod
    def _build_graph(self):
        pass

    @abc.abstractmethod
    def _train_func(self, batch_idx, batch_data):
        pass

    @abc.abstractmethod
    def _test_func(self, batch_idx, batch_data):
        pass

    def __create_writer(self):
        path = "SavedModel/Board/{}/".format(self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        return tf.summary.FileWriter(
            logdir=path,
            graph=self._graph,
            session=self._sess
        )

    def load(self):
        with self._graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.name)
            saver.restore(self._sess, path)
            print("Model \"{}\" loaded".format(self.name))

    def try_load(self):
        try:
            self.load()
        except:
            pass

    def close(self):
        self._sess.close()

    def save(self):
        with self._graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.name)
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            prefix = saver.save(self._sess, path)
            print("Model saved at \"{}\"".format(prefix))

    def __init_global_variables(self):
        print("Initializing global variables")
        self._sess.run(self.__init_op)

    def train(self, train_set, test_set, max_flip, max_epoch):
        self.__init_global_variables()

        flip_count = 0
        best_loss = None
        epoch = 0
        while epoch < max_epoch:
            batch_idx = 0
            for batch_data in train_set.fetch():
                self._train_func(batch_idx, batch_data)
                batch_idx += 1
            print()

            new_loss = self.__valid(test_set)
            if best_loss is None or new_loss < best_loss:
                best_loss = new_loss
                print("{} tested, new_loss={} best_loss={}".format(self.name, new_loss, best_loss))
                self.save()
            else:
                print("{} tested, new_loss={} best_loss={}".format(self.name, new_loss, best_loss))
                flip_count += 1
                if flip_count >= max_flip:
                    break

            epoch += 1
        print("Model '{}' train over".format(self.name))

    def __valid(self, test_set):
        total_loss = 0.0
        count = 0
        avg_loss = 0.0
        for batch_data in test_set.fetch():
            loss = self._test_func(count, batch_data)
            total_loss += loss
            count += 1
            avg_loss = total_loss / count
        print()
        return avg_loss
