import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

N = NeuronGroup(1000)
S = SynapseGroup(N, N)

weights = np.random.rand(N.size, N.size).astype(theano.config.floatX)
np.fill_diagonal(weights, 0.0)
S.connect(weights) # random connections

# TODO: scan + batch 20ms intervals onto the GPU
# http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
start = time.clock()
for now in range(0, 1000, 1):
    noise = np.random.rand(N.size).astype(theano.config.floatX) * 5.0
    spikes = N.tick(now, noise) # random thalmic noise
    S.tick(now, spikes, spikes)
    print("tick:", now, np.count_nonzero(spikes))

end = time.clock()
print("duration:", end - start)
