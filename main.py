import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

def main():
    duration = 1000.0

    N = NeuronGroup(1000)

    S = SynapseGroup(N, N)
    S.W.set_value(np.random.rand(N.size, N.size).astype(theano.config.floatX) * 20.0 - 10.0)

    # TODO: scan + batch 20ms intervals onto the GPU
    # http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
    start = time.clock()
    for now in np.arange(0.0, duration, 1.0):
        spikes = N.tick(now, np.random.rand(N.size).astype(theano.config.floatX) * 5.0)
        S.tick(now, spikes, spikes)

        weights = S.W.get_value()

        end = time.clock()
        print ("t", now, "real", end - start, "rate", np.count_nonzero(spikes), "min", weights.min(), "max",  weights.max(), "std", weights.std())

if __name__ == "__main__":
    main()
