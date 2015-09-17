import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

duration = 20.0

N = NeuronGroup(1000)
S = SynapseGroup(N, N)

weights = np.ones((N.size, N.size)).astype(theano.config.floatX) * 5.0
np.fill_diagonal(weights, 0.0)
S.W.set_value(weights)

# TODO: scan + batch 20ms intervals onto the GPU
# http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
start = time.clock()
for now in np.arange(0.0, duration, 1.0):
    spikes = N.tick(now, np.ones(N.size).astype(theano.config.floatX) * 4.0)
    S.tick(now, spikes, spikes)
    print (now, np.count_nonzero(spikes))

end = time.clock()
print("duration:", end - start)
