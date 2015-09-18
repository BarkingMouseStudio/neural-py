import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

def main():
    duration = 1000.0

    N1 = NeuronGroup(500)
    N2 = NeuronGroup(2000)

    S_exc = SynapseGroup(N1, N2)
    S_exc.W_max = 10
    S_exc.W_min = 0
    S_exc.W.set_value(np.random.rand(N2.size, N1.size).astype(theano.config.floatX))

    S_inh = SynapseGroup(N1, N2)
    S_inh.W_max = 0
    S_inh.W_min = -10
    S_inh.W.set_value(np.random.rand(N2.size, N1.size).astype(theano.config.floatX) * -1.0)

    # TODO: scan + batch 20ms intervals onto the GPU
    # http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
    start = time.clock()
    for now in np.arange(0.0, duration, 1.0):
        spikes1 = N1.tick(now, np.random.rand(N1.size).astype(theano.config.floatX) * 6.0)
        spikes2 = N2.tick(now, np.zeros(N2.size, dtype=theano.config.floatX))

        S_exc.tick(now, spikes1, spikes2)
        S_inh.tick(now, spikes1, spikes2)

        print (now, np.count_nonzero(spikes1), np.count_nonzero(spikes2))

    end = time.clock()
    print("duration:", end - start)

if __name__ == "__main__":
    main()
