import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N1, N2, weight_min=-10.0, weight_max=10.0):
        self.N1 = N1
        self.N2 = N2
        self.scheduler = N2.scheduler
        self.delay = 1

        a_sym = 0.05
        tau_a = 10.0
        tau_b = 10.0

        self.weight = weight = theano.shared(np.zeros((N2.size, N1.size), dtype=floatX), name="weight", borrow=True)
        pre_t = theano.shared(np.zeros((N2.size, N1.size), dtype=floatX), name="pre_t", borrow=True)
        post_t = theano.shared(np.zeros((N1.size, N2.size), dtype=floatX), name="post_t", borrow=True)

        dt = post_t - pre_t.T
        dw = a_sym * (1.0 - (dt / tau_a)**2.0) * T.exp(-T.abs_(dt) / tau_b)

        now = T.scalar("now")
        spikes = T.vector("spikes")

        self.pre_recv = theano.function([now, spikes], [pre_t],
            updates=[(pre_t, T.switch(spikes, now, pre_t))], name="pre_recv")

        self.post_recv = theano.function([now, spikes], [post_t],
            updates=[(post_t, T.switch(spikes, now, post_t))], name="post_recv")

        # integrate dw for dt's within 50ms window, then clamp
        # NOTE: even though both sides of the switch are calculated, this is still faster. cannot use ifelse because it is not element-wise.
        self.integrate = theano.function([], weight,
            updates=[(weight, T.clip(weight + dw, weight_min, weight_max))], name="integrate")

        self.apply_spikes = theano.function([spikes],
            T.sum(weight.T * spikes, axis=1, dtype=floatX, acc_dtype=floatX), name="apply_spikes")

    def tick(self, now, learning_enabled=True, transmission_enabled=True):
        if learning_enabled:
            # for incoming neurons that spiked, update their synapses
            self.pre_recv(now, self.N1.spikes)

            # for the receiving neurons that spiked, update their synapses
            self.post_recv(now, self.N2.spikes)

            # integrate with new pre/post times
            self.integrate()

        # TODO: transmit spikes but do not override training inputs (so we can train hidden layers)
        if transmission_enabled:
            # convert neuron spikes into their respective outgoing synaptic weights and delays
            spikes_out = self.apply_spikes(self.N1.spikes)

            # schedule those spikes
            t = now + self.delay
            self.scheduler.apply_schedule(t, spikes_out)
