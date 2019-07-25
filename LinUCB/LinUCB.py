# LinUCB algorithm
import numpy as np
import math

# class LinUCB
# usage: test = LinUCB(alpha, k, p)
#        LinUCB.regret(step, theta, cont, noise): get the regret of the function after t step
#        LinUCB.update_alpha(alpha): update the alpha for the simulation
# # # alpha : the exploration alpha
# # # k : choose from k actions
# # # p : the dimension of the Ct vector
# # # step : the algorithm go for step steps
# # # theta : the optimized theta for simulation( the dimension is defined by the alogorithm , for now is  k*p)
# # # cont : [mu, sigma] a 2 dimensional vector for mu and sigma in gauss-context distribution
# # # noise : [mu, sigma] a 2 dimensional vector for mu and sigma in gause-noise distribution

class LinUCB:
    def __init__(self, env):
        # initialize
        self.env = env
        self.alpha = env.get_alpha()
        self.k = env.get_k()
        self.p = env.get_p()
        self.d = self.k * self.p

        # for calculation
        self.A = np.eye(self.d)
        self.A_inv = np.eye(self.d)
        self.b = np.zeros((self.d, 1))

        # for result
        self._theta = 0
        
    # update_alpha in the run
    def update_alpha(self, alpha):
        self.alpha = alpha

    # update_Ab
    def update_Ab(self, f_t, r_t):
        self.A = self.A + np.dot(np.transpose(f_t), f_t)
        self.b = self.b + r_t * np.transpose(f_t)

    def update_b(self, f_t, r_t):
        self.b = self.b + r_t * np.transpose(f_t)

    # k_features: observe cont and convert to features
    # mapfunction: [ 0, 0, ... , Context_t, ... ,0]
    #                            -------
    #                               k
    def observe_features(self):
        cont = self.env.get_cont()
        k_features = []
        mu = np.array([1] * self.p)
        sigma = np.eye(self.p)
        context = np.random.multivariate_normal(mu, sigma, 1)
        context = context[0]
        # context = np.random.normal(cont[0], cont[1], self.p)
        for i in range(0, self.k):
            # convert context to features
            feature = np.array([0] * i * self.p + context.tolist() + [0] * (self.k - i - 1) * self.p)
            feature = feature.reshape(1, self.p * self.k)
            k_features.append(feature)
        return k_features
    
    # reward_by_simulation
    # variable:
    # # # outside:f_t, theta, noise
    def observe_payoff(self, f_t, theta):
        noise = self.env.get_noise()
        r_t = np.dot(f_t, theta) + np.random.normal(noise[0], noise[1])
        return r_t

    def observe_delay(self, f_t):
        norm2 = np.dot(f_t, np.transpose(f_t))
        delay_max = self.env.delay_max
        # return 0
        if norm2 < self.d * 0.5:
            return 0
        elif norm2 < self.d:
            return 0.25 * delay_max
        else:
            return delay_max

    # index : index of the action from k actions
    # variable:
    # # # outside:k_features, theta_t, self.A, self.alpha
    # # # inside: p_ta, a_t, index
    def argmax_pta(self, k_features, theta_t):
        p_ta = np.zeros(self.k)
        for f_t in range(0, self.k):
            # p_ta = f * theta_t + temp
            # temp = alpha * \sqrt{f * A^(-1) * f^(T)}
            f = k_features[f_t]
            temp = np.dot(np.dot(f, self.A_inv), np.transpose(f))
            p_ta[f_t] = np.dot(f, theta_t) + self.alpha * math.sqrt(temp)
        return np.argmax(p_ta)

    # max_reward_simulation
    def max_rt(self, k_features, theta):
        reward = np.zeros((1, self.k))
        for f_t in range(0, self.k):
            reward[0][f_t] = np.dot(k_features[f_t], theta)
        return np.max(reward)

    def regret_t(self, step, theta_star):
        regret_t = np.zeros((1, step))
        total_regret = 0
        # run
        for t in range(0, step):
            self.A_inv = np.linalg.inv(self.A)
            theta_t = np.dot(self.A_inv, self.b)  # theta_t = A^(-1) * b
            self._theta = theta_t
            k_features = self.observe_features()
            a_t = self.argmax_pta(k_features, theta_t)
            best_action = k_features[a_t]
            r_t = self.observe_payoff(best_action, theta_star)
            total_regret += self.max_rt(k_features, theta_star) - np.dot(best_action, theta_star)
            regret_t[0][t] = total_regret[0]
            self.update_Ab(best_action, r_t)
        return regret_t

    def regret_delay_t(self, step, theta_star):
        regret_t = np.zeros((1, step))
        total_regret = 0
        self.env.clear_delay()
        for t in range(0, step):
            self.A_inv = np.linalg.inv(self.A)
            theta_t = np.dot(self.A_inv, self.b)  # theta_t = A^(-1) * b
            self._theta = theta_t
            k_features = self.observe_features()
            a_t = self.argmax_pta(k_features, theta_t)
            best_action = k_features[a_t]
            r_t = self.observe_payoff(best_action, theta_star)
            total_regret += self.max_rt(k_features, theta_star) - np.dot(best_action, theta_star)
            regret_t[0][t] = total_regret[0]

            delay = self.observe_delay(best_action)
            if delay == 0:
                self.update_Ab(best_action, r_t)
                # print("non_delay", t)
            else:
                self.env.store_delay(best_action, r_t, delay + t)
                # print("delay", t, delay+t)

            get_delay = self.env.observe_delay(t)
            while get_delay.__len__() > 1:
                self.update_Ab(get_delay[0], get_delay[1])
                get_delay = self.env.observe_delay(t)
                # print("get_delay", t)

        return regret_t



