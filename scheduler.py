import numpy as np
import theano
import theano.tensor as T

class Scheduler:
    def __init__(self, N_size, max_delay=20):
        self.max_delay = max_delay
        self.schedule = schedule = theano.shared(np.zeros((N_size, max_delay)).astype(theano.config.floatX), name="schedule")

        t = T.iscalar("t")
        spikes = T.vector("spikes")

        self.apply_schedule = theano.function([t, spikes], schedule,
            updates=[(schedule, T.inc_subtensor(schedule.T[t % max_delay], spikes).T)], name="apply_schedule")

        self.get_schedule = theano.function([t], schedule.T[t % max_delay], name="get_schedule")

    def apply(self, t, spikes):
        self.apply_schedule(t, spikes)

    def get(self, t):
        return self.get_schedule(t)
