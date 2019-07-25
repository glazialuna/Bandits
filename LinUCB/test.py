from Plot import Plot
from simulation import Environment

#  simulation and plot
#  env = Environment()
# alpha = 1.1
delta = 0.1
k = 3
p = 5
steps = 2000
runs = 100
cont = [0, 1]
noise = [0, 1]
# # theta_star = env.set_theta(k * p, 1)
# runs = 1
delay_max = 20

simulation = Environment(k, p, delta, cont, noise, delay_max)
# a = Plot(simulation, runs, steps).plot_regret_t()
a = Plot(simulation, runs, steps).plot_delay()