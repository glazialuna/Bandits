# simulation
import numpy as np
# Environment
# Usage: env = (self, k, p, alpha, cont, noise, delay_max)
# for the environment_set

class Environment:
    def __init__(self, k, p, delta, cont, noise, delay_max):
        self.k = k
        self.p = p
        self.alpha = 1 + np.sqrt(0.5 * np.log(2/delta))
        self.cont = cont
        self.noise = noise

        self.delay_max = delay_max
        self.delay_index = [0] * delay_max
        self.delay_acts = [0] * delay_max
        self.delay_rt = [0] * delay_max

    # generate theta of dimentsion r
    # ||theta|| <= norm2
    def set_theta(self):
        theta = np.random.randn(self.k * self.p, 1)
        return theta / np.linalg.norm(theta, ord=2)

    def get_k(self):
        return self.k

    def get_p(self):
        return self.p

    def get_cont(self):
        return self.cont

    def get_noise(self):
        return self.noise

    def get_alpha(self):
        return self.alpha

    def get_delay_max(self):
        return self.delay_max

    def clear_delay(self):
        self.delay_index = [0] * self.delay_max

    def store_delay(self, best_action, r_t, delay):
        for i in range(self.delay_max):
            if self.delay_index[i] == 0:
                self.delay_index[i] = delay + 1
                self.delay_acts[i] = best_action
                self.delay_rt[i] = r_t
            return

    def observe_delay(self, i):
        if i == 0:
            return [-1]
        get_delay = i % self.delay_max + 1
        for i in range(self.delay_max):
            if self.delay_index[i] == get_delay:
                self.delay_index[i] = 0
                return [self.delay_acts[i], self.delay_rt[i]]
        return [-1]

