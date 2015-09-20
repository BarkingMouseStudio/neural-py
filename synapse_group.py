import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N1, N2):
        self.scheduler = N2.scheduler
        self.delay = 1

        self.transmission_enabled = True
        self.learning_enabled = True

        W_min = -10.0
        W_max = 10.0
        a_sym = 0.05
        tau_a = 10.0
        tau_b = 10.0

        self.W = W = theano.shared(np.zeros((N2.size, N1.size), dtype=floatX), name="W", borrow=True)
        pre_t = theano.shared(np.zeros((N2.size, N1.size), dtype=floatX), name="pre_t", borrow=True)
        post_t = theano.shared(np.zeros((N1.size, N2.size), dtype=floatX), name="post_t", borrow=True)

        dt = post_t.T - pre_t
        dw = a_sym * (1.0 - (dt / tau_a)**2.0) * T.exp(-T.abs_(dt) / tau_b)

        now = T.scalar("now")
        spikes = T.vector("spikes")

        self.pre_recv = theano.function([now, spikes], [pre_t],
            updates=[(pre_t, T.switch(spikes, now, pre_t))], name="pre_recv")

        self.post_recv = theano.function([now, spikes], [post_t],
            updates=[(post_t, T.switch(spikes, now, post_t))], name="post_recv")

        # integrate dw for dt's within 50ms window, then clamp
        # NOTE: even though both sides of the switch are calculated, this is still faster. cannot use ifelse because it is not element-wise.
        self.integrate = theano.function([], W,
            updates=[(W, T.clip(T.switch(T.lt(T.abs_(dt), 50.0), W + dw, W), W_min, W_max))], name="integrate")

        self.apply_spikes = theano.function([spikes],
            T.sum(W * spikes, axis=1, dtype=floatX, acc_dtype=floatX), name="apply_spikes")

    def set_training(self, is_training):
        if is_training:
            self.learning_enabled = True
            self.transmission_enabled = False
        else:
            self.learning_enabled = False
            self.transmission_enabled = True

    def tick(self, now, spikes_1, spikes_2):
        if self.learning_enabled:
            # for incoming neurons that spiked, update their synapses
            self.pre_recv(now, spikes_1)

            # for the receiving neurons that spiked, update their synapses
            self.post_recv(now, spikes_2)

            # integrate with new pre/post times
            self.integrate()

        # TODO: transmit spikes but do not override training inputs (so we can train hidden layers)
        if self.transmission_enabled:
            # convert neuron spikes into their respective outgoing synaptic weights and delays
            spikes_out = self.apply_spikes(spikes_1)

            # schedule those spikes
            t = now + self.delay
            self.scheduler.apply_schedule(t, spikes_out)
