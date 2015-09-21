import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

floatX = theano.config.floatX

# forces a neuron group to fire at a particular rate using a poisson process
def main():
    duration = 1000.0
    N = NeuronGroup(10)

    rate = 500.0 # Hz
    dt = 1.0 / 1000.0 # 1 ms
    rate_ms = rate * dt

    for now in np.arange(0.0, duration, 1.0):
        input_rate = (np.random.rand(N.size) < rate_ms).astype(floatX) * 125.0
        spikes = N.tick(now, input_rate)
        print (now, N.rate.get_value().mean())

if __name__ == "__main__":
    main()
