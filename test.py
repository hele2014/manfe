from db import *
from detnet import *
from manfe import *


class HyperParameters:
    def __init__(self):
        self.snr = 25
        self.rho = 0.5
        self.alpha = 1.5
        self.beta = 0.0
        self.mu = 0.0
        self.sigma = 1.0

        self.max_flip = 1
        self.max_epoch = 100
        self.train_total_batch = 10000
        self.valid_total_batch = 2000
        self.test_total_batch = 1000


def train(hps):
    train_set = TrainDb(hps)
    valid_set = ValidDb(hps)

    model = MANFE(hps.alpha, 8, 4)
    model.train(train_set, valid_set, hps.max_flip, hps.max_epoch)
    model.close()


def benchmark(hps):
    # detnet = DetNet(NUM_ANT, 4 * 2 * NUM_ANT, 30, hps.alpha, hps.snr)
    # detnet.load()

    manfe = MANFE(hps.alpha, 8, 4)
    manfe.load()

    t = 0
    err_detnet = 0
    err_gamp = 0
    err_mld = 0
    err_gamp_mld_1 = 0
    err_gamp_mld_2 = 0
    err_gamp_manfe_1 = 0
    err_gamp_manfe_2 = 0
    err_manfe = 0
    total_bits = 0

    test_set = TestDb(hps)

    print("BER benchmark SNR:{} alpha:{}".format(hps.snr, hps.alpha))

    batch_count = 0
    for y, h, s, w in test_set.fetch():
        batch_count += 1

        bits = get_bits(s)
        total_bits += bits.size

        # bits_detnet = detnet.detect_bits(y, h)
        s_gamp, t_ = amp_batch(y, h, loop=30)
        t += t_
        # bits_gamp = get_bits(s_gamp)
        # bits_gamp_mld_1 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=1, use_mld=True)
        # bits_gamp_mld_2 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=2, use_mld=True)
        # bits_gamp_manfe_1 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=1, use_mld=False)
        # bits_gamp_manfe_2 = manfe.detect_bits_with_initial_guess(y, h, s_gamp, max_error_symbols=3, use_mld=False)
        bits_mld = maximum_likelihood_detect_bits(y, h)
        bits_manfe = manfe.detect_bits(y, h)

        # err_detnet += check_wrong_bits(bits, bits_detnet)
        # err_gamp += check_wrong_bits(bits, bits_gamp)
        # err_gamp_mld_1 += check_wrong_bits(bits, bits_gamp_mld_1)
        # err_gamp_mld_2 += check_wrong_bits(bits, bits_gamp_mld_2)
        # err_gamp_manfe_1 += check_wrong_bits(bits, bits_gamp_manfe_1)
        # err_gamp_manfe_2 += check_wrong_bits(bits, bits_gamp_manfe_2)
        err_mld += check_wrong_bits(bits, bits_mld)
        err_manfe += check_wrong_bits(bits, bits_manfe)

        ber_detnet = err_detnet / total_bits
        ber_gamp = err_gamp / total_bits
        ber_gamp_mld_1 = err_gamp_mld_1 / total_bits
        ber_gamp_mld_2 = err_gamp_mld_2 / total_bits
        ber_gamp_manfe_1 = err_gamp_manfe_1 / total_bits
        ber_gamp_manfe_2 = err_gamp_manfe_2 / total_bits
        ber_mld = err_mld / total_bits
        ber_manfe = err_manfe / total_bits

        precision = 1 / total_bits

        data_text = "MLD:{:e} MANFE:{:e} ({:e} {})".format(
            # ber_detnet,
            # ber_gamp,
            # ber_gamp_mld_1,
            # ber_gamp_mld_2,
            # ber_gamp_manfe_1,
            # ber_gamp_manfe_2,
            ber_mld,
            ber_manfe,
            precision,
            batch_count)

        print(data_text, end="\r")

        if ber_gamp_manfe_1/precision >= 1000:
            break

    print()
    print()
    # detnet.close()
    manfe.close()


def main():
    ghps = HyperParameters()

    for i in [1.9]:
        ghps.alpha = i
        for j in [25]:
            ghps.snr = j
            print("snr={} alpha={}".format(ghps.snr, ghps.alpha))
            # train(ghps)
            benchmark(ghps)
            print()


if __name__ == "__main__":
    main()
