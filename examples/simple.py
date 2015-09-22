import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import theano
import theano.tensor as T

from neural.neuron_group import NeuronGroup
from neural.synapse_group import SynapseGroup

floatX = theano.config.floatX

def main():
    duration = 1000.0

    N = NeuronGroup(1000)

    S = SynapseGroup(N, N)
    S.W.set_value(np.random.rand(N.size, N.size).astype(floatX) * 20.0 - 10.0)

    # TODO: scan + batch 20ms intervals onto the GPU
    # http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
    start = time.clock()
    for now in np.arange(0.0, duration, 1.0):
        N.tick(now, np.random.rand(N.size).astype(floatX) * 5.0)
        S.tick(now)

        weights = S.W.get_value()

        end = time.clock()
        print ("t", now, "real", end - start, "rate", np.count_nonzero(N.spikes))

if __name__ == "__main__":
    main()
