# Plot the runs result
import numpy as np
from LinUCB import LinUCB
from simulation import Environment
import matplotlib.pyplot as pt
from tqdm import tqdm as tq
import profile
# from numpy import numba
# @numba.autojit

# class Plot
# usage: test = Plot()
#        test.plot_regrets(runs,steps,model,theta, cont, noise): a figure
# # # steps: a 3 dimensional vector [step_begin, step_end, step_gap ]
# # # runs: the runs count
# # # model: model class


class Plot:
    def __init__(self, env,  runs, steps):
        self.env = env
        self.runs = runs
        self.steps = steps
        self.step_list = np.arange(0, steps)

    # Plot the regret-t with self.runs times
    # # subPlot the mean for the times
    def plot_regret_t(self):
        pt.figure()
        mean = np.zeros(self.steps)
        for i in tq(range(0, self.runs)):
            np.random.seed(i)
            theta_star = self.env.set_theta()
            r = LinUCB(self.env).regret_t(self.steps, theta_star)
            pt.subplot(211)
            pt.plot(self.step_list, r[0])
            mean += r[0]
        pt.subplot(212)
        mean = [i/self.runs for i in mean]
        pt.plot(self.step_list, mean)
        pt.show()

    # Plot the regret-t with self.runs times
    # # subPlot the mean for the times
    def plot_delay(self):
        pt.figure()
        mean1 = np.zeros(self.steps)
        mean2 = np.zeros(self.steps)
        for i in tq(range(0, self.runs)):
            np.random.seed(i)
            theta_star = self.env.set_theta()
            r1 = LinUCB(self.env).regret_delay_t(self.steps, theta_star)
            r2 = LinUCB(self.env).regret_t(self.steps, theta_star)
            # pt.subplot(221)
            # pt.plot(self.step_list, r[0])
            mean1 += r1[0]
            mean2 += r2[0]
        # pt.subplot(222)
        mean1 = [i/self.runs for i in mean1]
        mean2 = [i/self.runs for i in mean2]

        # for i in tq(range(0, self.runs)):
        #     np.random.seed(i)
        #     theta_star = self.env.set_theta()
        #     # r1 = LinUCB(self.env).regret_delay_t(self.steps, theta_star)
        #     r2 = LinUCB(self.env).regret_t(self.steps, theta_star)
        #     # pt.subplot(221)
        #     # pt.plot(self.step_list, r[0])
        #     # mean1 += r1[0]
        #     mean2 += r2[0]
        # # pt.subplot(222)
        # # mean1 = [i/self.runs for i in mean1]
        # mean2 = [i/self.runs for i in mean2]

        pt.plot(self.step_list, mean1, label="delay")
        pt.plot(self.step_list, mean2, label="non_delay")
        pt.legend()
        pt.show()