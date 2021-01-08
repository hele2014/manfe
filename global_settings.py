import itertools

import numpy as np

NUM_ANT = 4

TIME_SLOTS_PER_PACKET = 10
PACKETS_PER_BATCH = 100

TIMES_SLOTS_PER_BATCH = PACKETS_PER_BATCH * TIME_SLOTS_PER_PACKET
PACKET_SIZE = TIME_SLOTS_PER_PACKET * (2 * NUM_ANT)

if NUM_ANT <= 8:
    QPSK_BITS_PER_SYMBOL = 2
    QPSK_CANDIDATE_SIZE = 2 ** (2 * NUM_ANT)
    QPSK_CANDIDATES = np.array([x for x in itertools.product([1, -1], repeat=2 * NUM_ANT)]).T / np.sqrt(2)
