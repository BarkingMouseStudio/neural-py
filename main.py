import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup

max_delay = 20
N1_size = 1000
N2_size = 1000

scheduler_1 = theano.shared(np.zeros((N1_size, max_delay)).astype(theano.config.floatX), name="scheduler_1")
scheduler_2 = theano.shared(np.zeros((N2_size, max_delay)).astype(theano.config.floatX), name="scheduler_2")

N1 = NeuronGroup(N1_size, scheduler_1, max_delay)
N2 = NeuronGroup(N2_size, scheduler_2, max_delay)
S = SynapseGroup(N1_size, N2_size, scheduler_2, max_delay)

weights = np.random.rand(N1_size, N2_size).astype(theano.config.floatX)
np.fill_diagonal(weights, 0.0)
S.connect(weights) # random connections

# TODO: scan + batch 20ms intervals onto the GPU
# http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
start = time.clock()
for now in range(0, 1000, 1):
    spikes_1 = N1.tick(now, np.random.rand(N1_size).astype(theano.config.floatX) * 5.0) # random thalmic noise
    spikes_2 = N2.tick(now, np.zeros(N2_size).astype(theano.config.floatX) * 120.0) # random thalmic noise
    S.tick(now, spikes_1, spikes_2)
    print("tick:", now)

end = time.clock()
print("duration:", end - start)
