import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class SynapseGroup:
    # scheduler should be receiving scheduler
    def __init__(self, N_pre, N_post, weight_min=-10.0, weight_max=10.0):
        self.N_pre = N_pre
        self.N_post = N_post
        self.scheduler = N_post.scheduler
        self.delay = 1

        a_sym = 0.05
        tau_a = 10.0
        tau_b = 10.0

        self.weight = weight = theano.shared(np.zeros((N_post.size, N_pre.size), dtype=floatX), name="weight", borrow=True)
        t_pre = theano.shared(np.zeros(N_pre.size, dtype=floatX), name="t_pre", borrow=True)
        t_post = theano.shared(np.zeros(N_post.size, dtype=floatX), name="t_post", borrow=True)

        dt = t_post - t_pre.dimshuffle(0, 'x')
        dw = a_sym * (1.0 - (dt / tau_a)**2.0) * T.exp(-T.abs_(dt) / tau_b)

        now = T.scalar("now")
        spikes_pre = T.vector("spikes_pre")
        spikes_post = T.vector("spikes_post")
        spikes_merged = T.matrix("spikes_merged")

        merge_spikes = spikes_pre.dimshuffle(0, 'x') + spikes_post
        weight_cond = T.gt(merge_spikes, 0.0)
        weight_update = T.switch(weight_cond, weight + dw, weight)
        weight_clamp = T.clip(weight_update, weight_min, weight_max)

        self.pre_recv = theano.function([now, spikes_pre], t_pre,
            updates=[(t_pre, T.switch(spikes_pre, now, t_pre))], name="pre_recv")

        self.post_recv = theano.function([now, spikes_post], t_post,
            updates=[(t_post, T.switch(spikes_post, now, t_post))], name="post_recv")

        # integrate dw for dt's within 50ms window, then clamp
        # NOTE: even though both sides of the switch are calculated, this is still faster. cannot use ifelse because it is not element-wise.
        self.integrate = theano.function([spikes_pre, spikes_post], weight,
            updates=[(weight, weight_clamp)], name="integrate")

        self.apply_spikes = theano.function([spikes_pre],
            T.sum(weight.T * spikes_pre, axis=1, dtype=floatX, acc_dtype=floatX), name="apply_spikes")

    def tick(self, now, learning_enabled=True, transmission_enabled=True):
        if learning_enabled:
            # for incoming neurons that spiked, update their synapses
            self.pre_recv(now, self.N_pre.spikes)

            # for the receiving neurons that spiked, update their synapses
            self.post_recv(now, self.N_post.spikes)

            # integrate with new pre/post times
            self.integrate(self.N_pre.spikes, self.N_post.spikes)

        # TODO: transmit spikes but do not override training inputs (so we can train hidden layers)
        if transmission_enabled:
            # convert neuron spikes into their respective outgoing synaptic weights and delays
            spikes_out = self.apply_spikes(self.N_pre.spikes)

            # schedule those spikes
            t = now + self.delay
            self.scheduler.apply_schedule(t, spikes_out)
