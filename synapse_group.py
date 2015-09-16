import numpy as np
import theano
import theano.tensor as T

def inspect_inputs(i, node, fn):
    print(i, node, "inputs:\n\t", [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print(i, node, "outputs:\n\t", [output[0] for output in fn.outputs])

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N1_size, N2_size, scheduler):
        self.scheduler = scheduler
        self.delay = 1

        W_min = -10.0
        W_max = 10.0
        a_sym = 0.05
        tau_a = 10.0
        tau_b = 10.0

        self.W = W = theano.shared(np.zeros((N1_size, N2_size)), name="W")
        pre_t = theano.shared(np.zeros((N1_size, N2_size)), name="pre_t")
        post_t = theano.shared(np.zeros((N1_size, N2_size)), name="post_t")

        dt = post_t - pre_t
        dw = a_sym * (1.0 - (dt / tau_a)**2) * T.exp(-T.abs_(dt) / tau_b)

        now = T.iscalar("now")
        spikes = T.vector("spikes")

        self.pre_recv_now = theano.function([now, spikes], [pre_t],
            updates=[(pre_t, T.switch(spikes, now, pre_t.T).T)], name="pre_recv_now")

        self.post_recv_now = theano.function([now, spikes], [post_t],
            updates=[(post_t, T.switch(spikes, now, post_t))], name="post_recv_now")

        self.integrate = theano.function([], W,
            updates=[(W, T.clip(W + dw, W_min, W_max))], name="integrate")

        self.apply_spikes = theano.function([spikes],
            T.sum(W * spikes, axis=1, dtype=theano.config.floatX), name="apply_spikes")

    def connect(self, W_connect):
        self.W.set_value(W_connect)

    def tick(self, now, spikes_1, spikes_2):
        # for incoming neurons that spiked, update their synapses
        self.pre_recv_now(now, spikes_1)

        # for the receiving neurons that spiked, update their synapses
        self.post_recv_now(now, spikes_2)

        # integrate with new pre/post times
        self.integrate()

        # convert neuron spikes into their respective outgoing synaptic weights and delays
        spikes_out = self.apply_spikes(spikes_1)

        # schedule those spikes
        t = now + self.delay
        self.scheduler.apply_schedule(t, spikes_out)
