import numpy as np
import theano
import theano.tensor as T

class Scheduler:
    def __init__(self, N_size, max_delay=20):
        self.max_delay = max_delay
        self.schedule = schedule = theano.shared(np.zeros((max_delay, N_size)).astype(theano.config.floatX), name="schedule")

        t = T.iscalar("t")
        spikes = T.vector("spikes")

        self.apply_schedule = theano.function([t, spikes], schedule,
            updates=[(schedule, T.inc_subtensor(schedule[t % max_delay], spikes))], name="apply_schedule")

        self.get_schedule = theano.function([t], schedule[t % max_delay], name="get_schedule")

        self.clear_schedule = theano.function([t], schedule, updates=[
            (schedule, T.set_subtensor(schedule[t % max_delay], 0.0))
        ], name="clear_schedule")
