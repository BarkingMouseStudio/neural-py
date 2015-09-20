from __future__ import division

import time
import math
import numpy as np
import theano
import theano.tensor as T

from neuron_group import NeuronGroup
from synapse_group import SynapseGroup
from com_estimator import COMEstimator

floatX = theano.config.floatX

def normalize(x, mn, mx):
    return (x - mn) / (mx - mn)

def denormalize(x, mn, mx):
    return (x * (mx - mn)) + mn

def main():
    duration = 3.14 * 1000.0
    N = NeuronGroup(100)

    dt = 1.0 / 1000.0 # 1 ms
    noise_rate_ms = 3.0 * dt

    encoding = COMEstimator(N.size, 3.0, 1.0)

    for now in np.arange(0.0, duration, 1.0):
        input_value = math.sin(now / 1000.0)
        input_norm = normalize(input_value, -1.0, 1.0)
        input_rate = (np.random.rand(N.size) < noise_rate_ms).astype(floatX) * 125.0

        encoding.encode(input_rate, input_norm)

        N.tick(now, input_rate)

        output_norm = encoding.decode(N.rate.get_value())
        output_value = denormalize(output_norm, -1.0, 1.0)
        err = abs(input_norm - output_norm)
        print (now, err, input_value, output_value)

if __name__ == "__main__":
    main()
