import numpy as np
import theano
import theano.tensor as T

def inspect_inputs(i, node, fn):
    print(i, node, "inputs:\n\t", [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print(i, node, "outputs:\n\t", [output[0] for output in fn.outputs])

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N1, N2):
        self.scheduler = N2.scheduler
        self.delay = 1

        W_min = -10.0
        W_max = 10.0
        a_sym = 0.05
        tau_a = 10.0
        tau_b = 10.0

        self.W = W = theano.shared(np.zeros((N1.size, N2.size)).astype(theano.config.floatX), name="W")
        pre_t = theano.shared(np.zeros((N1.size, N2.size)).astype(theano.config.floatX), name="pre_t")
        post_t = theano.shared(np.zeros((N1.size, N2.size)).astype(theano.config.floatX), name="post_t")

        dt = T.cast(post_t - pre_t, theano.config.floatX)
        dw = T.cast(a_sym * (1.0 - (dt / tau_a)**2) * T.exp(-T.abs_(dt) / tau_b), theano.config.floatX)

        now = T.scalar("now")
        spikes1 = T.vector("spikes", dtype=theano.config.floatX)
        spikes2 = T.vector("spikes", dtype=theano.config.floatX)

        self.pre_recv_now = theano.function([now, spikes1], [pre_t],
            updates=[(pre_t, T.switch(spikes1, now, pre_t.T).T)], name="pre_recv_now")

        self.post_recv_now = theano.function([now, spikes2], [post_t],
            updates=[(post_t, T.switch(spikes2, now, post_t))], name="post_recv_now")

        self.integrate = theano.function([], W,
            updates=[(W, T.clip(W + dw, W_min, W_max))], name="integrate")

        self.apply_spikes = theano.function([spikes2],
            T.sum(W.T * spikes2, axis=1, dtype=theano.config.floatX), name="apply_spikes")

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

        return spikes_out
