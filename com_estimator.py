from __future__ import division

import math
import random

# Center-Of-Mass Estimator
class COMEstimator:
    def __init__(self, size, sigma, bin_size):
        self.sigma2_2 = 2.0 * math.pow(sigma, 2.0)
        self.rate_max = 500.0
        self.v = 125.0

        self.size = size
        self.bin_size = bin_size # number of neurons in the bin

        self.width = ((sigma * 3.0) * bin_size) / self.size
        self.scale = 1.0 - (self.width * 2.0)

    # f : [0, 1] -> [1, N]
    def f(self, v):
        v *= self.scale
        v += self.width
        return round(v * self.size)

    # f_inv : [1, N] -> [0, 1]
    def f_inv(self, x):
        v = x / self.size
        v -= self.width
        v /= self.scale
        return v

    def bin(self, x):
        return round(x / self.bin_size)

    # express firing rate of neuron x when the normalized value v_0 is encoded
    def encode(self, inp, theta):
        dt = 1.0 / 1000.0 # 1 ms
        for i in range(self.size):
            rate = self.rate_max * math.exp(-1.0 * (math.pow(self.bin(i - self.f(theta)), 2.0) / self.sigma2_2))

            # add to input so we maintain noise levels
            inp[i] += self.v if random.random() < rate * dt else 0.0

    def decode(self, rate):
        num = 0.0
        for i in range(self.size):
            num += self.f_inv(i) * rate[i]

        den = 0.0
        for i in range(self.size):
            den += rate[i]

        # if we cannot divide by zero, go to middle of range (0.5)
        return (num / den) if den != 0.0 else 0.5
