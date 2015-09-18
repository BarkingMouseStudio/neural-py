import numpy as np
import theano
import theano.tensor as T

from scheduler import Scheduler

floatX = theano.config.floatX

class NeuronGroup:
    def __init__(self, size):
        self.scheduler = Scheduler(size)
        self.size = size

        v_peak = 30.0
        a = 0.1
        b = 0.2
        c = -65.0
        d = 2.0
        tau = 0.5

        self.v = v = theano.shared(np.full(size, c, dtype=floatX), name="v")
        self.u = u = theano.shared(np.full(size, b * c, dtype=floatX), name="u")
        self.I = I = theano.shared(np.zeros(size, dtype=floatX), name="I")

        dv = tau * (0.04 * (v * v) + (v * 5.0) + 140.0 - u + I)
        du = tau * (a * ((b * v) - u))

        now = T.iscalar("now")
        DC = T.vector("DC")
        spikes = T.vector("spikes")
        schedule = T.vector("schedule")

        self.recv = theano.function([DC, schedule], I, updates=[(I, I + DC + schedule)])
        self.tick_v = theano.function([], v, updates=[(v, v + dv)])
        self.tick_u = theano.function([], u, updates=[(u, u + du)])
        self.threshold = theano.function([], v >= v_peak)
        self.reset = theano.function([spikes], [v, u, I], updates=[
            (v, T.switch(spikes, c, v)),
            (u, T.switch(spikes, u + d, u)),
            (I, T.zeros_like(I)),
        ])

    def tick(self, now, DC):
        schedule = self.scheduler.get_schedule(now)
        self.scheduler.clear_schedule(now)
        self.recv(DC, schedule)

        self.tick_v()
        self.tick_u()

        self.tick_v()
        self.tick_u()

        spikes = self.threshold()
        self.reset(spikes)
        return spikes
