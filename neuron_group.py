import numpy as np
import theano
import theano.tensor as T

class NeuronGroup:
    def __init__(self, size, scheduler, max_delay):
        self.scheduler = scheduler
        self.max_delay = max_delay
        self.size = size

        v_peak = 30.0
        a = 0.1
        b = 0.2
        c = -65.0
        d = 2.0

        self.v = v = theano.shared(np.full(size, c), name="v")
        self.u = u = theano.shared(np.full(size, c * b), name="u")
        self.I = I = theano.shared(np.zeros(size), name="I")

        dv = 0.04 * (v * v) + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)

        now = T.iscalar("now")
        DC = T.vector("DC")
        spikes = T.vector("spikes")

        self.recv = theano.function([now, DC], I, updates=[(I, I + DC + scheduler.T[now])])
        self.tick_v = theano.function([], v, updates=[(v, v + 0.5 * dv)])
        self.tick_u = theano.function([], u, updates=[(u, u + du)])
        self.threshold = theano.function([], T.gt(v, v_peak))
        self.reset = theano.function([spikes], [v, u], updates=[
            (v, T.switch(spikes, c, v)),
            (u, T.switch(spikes, u + d, u)),
            (I, np.zeros(size)),
        ])

    def tick(self, now, DC):
        t = now % self.max_delay
        self.recv(t, DC)

        self.tick_v()
        self.tick_v()
        self.tick_u()

        spikes = self.threshold()
        self.reset(spikes)
        return spikes
