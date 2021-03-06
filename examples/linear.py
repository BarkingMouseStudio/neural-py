from __future__ import division

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import theano
import theano.tensor as T

from neural.neuron_group import NeuronGroup
from neural.synapse_group import SynapseGroup
from neural.com_estimator import COMEstimator

floatX = theano.config.floatX

def normalize(x, mn, mx):
    return (x - mn) / (mx - mn)

def denormalize(x, mn, mx):
    return (x * (mx - mn)) + mn

def main():
    training_duration = 3.14 * 1000.0 * 3
    testing_duration = 3.14 * 1000.0 * 1

    N1 = NeuronGroup(100)
    N2 = NeuronGroup(50)

    S1 = SynapseGroup(N1, N2)
    S2 = SynapseGroup(N1, N2)

    weights = np.random.rand(N1.size, N2.size).astype(floatX)
    weights = denormalize(weights, -4.0, 0.0)
    S1.weight.set_value(weights)

    weights = np.random.rand(N1.size, N2.size).astype(floatX)
    weights = denormalize(weights, 0.0, 4.0)
    S2.weight.set_value(weights)

    dt = 1.0 / 1000.0 # 1 ms
    noise_rate_ms = 5.0 * dt

    encoding = COMEstimator(N1.size, 3.0, 1.0)
    decoding = COMEstimator(N2.size, 3.0, 1.0)

    # training
    for now in np.arange(0.0, training_duration, 1.0):
        input_value = math.sin(now / 1000.0)
        input_norm = normalize(input_value, -1.0, 1.0)

        output_value = -input_value
        output_norm = normalize(output_value, -1.0, 1.0)

        input_rate = (np.random.rand(N1.size) < noise_rate_ms).astype(floatX) * 125.0
        output_rate = (np.random.rand(N2.size) < noise_rate_ms).astype(floatX) * 125.0

        encoding.encode(input_rate, input_norm)
        decoding.encode(output_rate, output_norm)

        N1.tick(now, input_rate)
        N2.tick(now, output_rate)

        S1.tick(now, True, False)
        S2.tick(now, True, False)

        output_norm_actual = decoding.decode(N2.rate.get_value())

        err = abs(output_norm - output_norm_actual)
        print ("training", now / training_duration, err)

    # testing
    mse = 0.0

    for now in np.arange(0.0, testing_duration, 1.0):
        input_value = math.sin(now / 1000.0)
        input_norm = normalize(input_value, -1.0, 1.0)

        input_rate = np.zeros(N1.size).astype(floatX)
        output_rate = np.zeros(N2.size).astype(floatX)

        encoding.encode(input_rate, input_norm)

        N1.tick(now, input_rate)
        N2.tick(now, output_rate)

        S1.tick(now, False, True)
        S2.tick(now, False, True)

        output_value = -input_value
        output_norm = normalize(output_value, -1.0, 1.0)

        output_norm_actual = decoding.decode(N2.rate.get_value())
        output_value_actual = denormalize(output_norm_actual, -1.0, 1.0)

        err = abs(output_norm - output_norm_actual)
        mse += math.pow(err, 2.0)
        print ("testing", now / testing_duration, err)

    mse /= testing_duration
    print ("mse", mse)

if __name__ == "__main__":
    main()
