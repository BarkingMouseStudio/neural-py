import numpy as np
import theano
import theano.tensor as T

from scheduler import Scheduler

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

        self.v = v = theano.shared(np.full(size, c).astype(theano.config.floatX), name="v")
        self.u = u = theano.shared(np.full(size, c * b).astype(theano.config.floatX), name="u")
        self.I = I = theano.shared(np.zeros(size).astype(theano.config.floatX), name="I")

        dv = T.cast(0.04 * (v * v) + (5.0 * v) + 140.0 - u + I, theano.config.floatX)
        du = T.cast(a * ((b * v) - u), theano.config.floatX)

        now = T.iscalar("now")
        DC = T.vector("DC")
        spikes = T.vector("spikes")
        schedule = T.vector("schedule")

        self.recv = theano.function([DC, schedule], I, updates=[(I, I + DC + schedule)])
        self.tick_v = theano.function([], v, updates=[(v, v + (tau * dv))])
        self.tick_u = theano.function([], u, updates=[(u, u + (tau * du))])
        self.threshold = theano.function([], v >= v_peak)
        self.reset = theano.function([spikes], [v, u, I], updates=[
            (v, T.switch(spikes, c, v)),
            (u, T.switch(spikes, u + d, u)),
            (I, I * 0.0),
        ])

    def tick(self, now, DC):
        schedule = self.scheduler.get_schedule(now)
        self.recv(DC, schedule)

        # tick once
        self.tick_v()
        self.tick_u()

        # tick twice
        self.tick_v()
        self.tick_u()

        spikes = self.threshold()
        self.reset(spikes)
        return spikes
