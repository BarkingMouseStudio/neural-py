import time
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup
from scheduler import Scheduler

N_size = 1000

scheduler = Scheduler(N_size)

N = NeuronGroup(N_size, scheduler)
S = SynapseGroup(N_size, N_size, scheduler)

weights = np.random.rand(N_size, N_size).astype(theano.config.floatX)
np.fill_diagonal(weights, 0.0)
S.connect(weights) # random connections

# TODO: scan + batch 20ms intervals onto the GPU
# http://www.deeplearning.net/tutorial/logreg.html#defining-a-loss-function
start = time.clock()
for now in range(0, 1000, 1):
    noise = np.random.rand(N_size).astype(theano.config.floatX) * 5.0
    spikes = N.tick(now, noise) # random thalmic noise
    S.tick(now, spikes, spikes)
    print("tick:", now, np.count_nonzero(spikes))

end = time.clock()
print("duration:", end - start)
