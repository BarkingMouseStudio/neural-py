import numpy as np
import theano
import theano.tensor as T

def inspect_inputs(i, node, fn):
    print(i, node, "inputs:\n\t", [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print(i, node, "outputs:\n\t", [output[0] for output in fn.outputs])

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N1_size, N2_size, scheduler, max_delay):
        self.scheduler = scheduler
        self.max_delay = max_delay
        self.delay = 1

        W_min = T.cast(-10.0, 'float32')
        W_max = T.cast(10.0, 'float32')
        a_sym = T.cast(0.05, 'float32')
        tau_a = T.cast(10.0, 'float32')
        tau_b = T.cast(10.0, 'float32')

        self.W = W = theano.shared(np.zeros((N1_size, N2_size)), name="W")
        self.pre_t = pre_t = theano.shared(np.zeros((N1_size, N2_size)), name="pre_t")
        self.post_t = post_t = theano.shared(np.zeros((N1_size, N2_size)), name="post_t")

        dt = post_t - pre_t
        dw = a_sym * (1.0 - (dt / tau_a)**2) * T.exp(-T.abs_(dt) / tau_b)

        now = T.iscalar("now")
        spikes_1 = T.vector("spikes_1")
        spikes_2 = T.vector("spikes_2")

        self.pre_recv_now = theano.function([now, spikes_1], [pre_t],
            updates=[(pre_t, T.switch(spikes_1, now, pre_t.T).T)], mode='FAST_RUN')

        self.post_recv_now = theano.function([now, spikes_2], [post_t],
            updates=[(post_t, T.switch(spikes_2, now, post_t))], mode='FAST_RUN')

        # NOTE: pre/post times should not overlap in a given tick from the same set of neurons
        self.integrate = theano.function([], W,
            updates=[(W, T.clip(W + dw, W_min, W_max))], mode='FAST_RUN')

        self.apply_spikes = theano.function([spikes_1],
            T.sum(W.T * spikes_1, axis=0, dtype=theano.config.floatX), mode='FAST_RUN')

        self.apply_scheduler = theano.function([now, spikes_1], scheduler,
            updates=[(scheduler, T.inc_subtensor(scheduler.T[now], spikes_1).T)], mode='FAST_RUN')

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
        t = (now + self.delay) % self.max_delay
        self.apply_scheduler(t, spikes_out)
