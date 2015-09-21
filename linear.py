from __future__ import division

import cProfile
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
    training_duration = 3.14 * 1000.0 * 3
    testing_duration = 3.14 * 1000.0 * 1

    N1 = NeuronGroup(100)
    N2 = NeuronGroup(100)

    S1 = SynapseGroup(N1, N2)
    S2 = SynapseGroup(N1, N2)

    weights = np.random.rand(N1.size, N2.size).astype(floatX)
    weights = denormalize(weights, -4.0, 0.0)
    S1.W.set_value(weights)

    weights = np.random.rand(N1.size, N2.size).astype(floatX)
    weights = denormalize(weights, 0.0, 4.0)
    S2.W.set_value(weights)

    dt = 1.0 / 1000.0 # 1 ms
    noise_rate_ms = 5.0 * dt

    encoding = COMEstimator(N1.size, 3.0, 1.0)

    S1.set_training(True)
    S2.set_training(True)

    # training
    for now in np.arange(0.0, training_duration, 1.0):
        input_value = math.sin(now / 1000.0)
        input_norm = normalize(input_value, -1.0, 1.0)

        output_value = -input_value
        output_norm = normalize(output_value, -1.0, 1.0)

        input_rate = (np.random.rand(N1.size) < noise_rate_ms).astype(floatX) * 125.0
        output_rate = (np.random.rand(N2.size) < noise_rate_ms).astype(floatX) * 125.0

        encoding.encode(input_rate, input_norm)
        encoding.encode(output_rate, output_norm)

        spikes1 = N1.tick(now, input_rate)
        spikes2 = N2.tick(now, output_rate)

        S1.tick(now, spikes1, spikes2)
        S2.tick(now, spikes1, spikes2)

        output_norm_actual = encoding.decode(N2.rate.get_value())

        err = abs(output_norm - output_norm_actual)
        print ("training", now / training_duration, err)

    # testing
    S1.set_training(False)
    S2.set_training(False)
    mse = 0.0

    for now in np.arange(0.0, testing_duration, 1.0):
        input_value = math.sin(now / 1000.0)
        input_norm = normalize(input_value, -1.0, 1.0)

        input_rate = np.zeros(N1.size).astype(floatX)
        output_rate = np.zeros(N2.size).astype(floatX)

        encoding.encode(input_rate, input_norm)

        spikes1 = N1.tick(now, input_rate)
        spikes2 = N2.tick(now, output_rate)

        S1.tick(now, spikes1, spikes2)
        S2.tick(now, spikes1, spikes2)

	output_value = -input_value
	output_norm = normalize(output_value, -1.0, 1.0)

        output_norm_actual = encoding.decode(N2.rate.get_value())
        output_value_actual = denormalize(output_norm_actual, -1.0, 1.0)

        err = abs(output_norm - output_norm_actual)
        mse += math.pow(err, 2.0)
        print ("testing", now / testing_duration, err)

    mse /= testing_duration
    print ("mse", mse)

if __name__ == "__main__":
    main()
