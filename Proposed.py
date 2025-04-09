import time
import numpy as np


# Improved Teamwork Optimization Algorithm (ITOA) Starting Line No. 22
def Proposed(agents, fobj, VRmin, VRmax, Max_iter):
    num_agents, dim = agents.shape[0], agents.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    best_agent = np.zeros((dim, 1))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        for i in range(num_agents):

            fitness = fobj(agents[i, :])

            currentfit = fitness
            bestfit = min(fitness)
            worstfit = max(fitness)
            meanfit = np.mean(fitness)
            rand_index = currentfit / (bestfit + worstfit + meanfit)
            partner = agents[rand_index]

            beta = np.random.uniform(0, 1)
            new_agent = agents[i] + beta * (best_agent - agents[i]) + (1 - beta) * (partner - agents[i])

            # Clamp new_agent within bounds
            new_agent = np.clip(new_agent, lb, ub)

            if fobj(new_agent) < fobj(agents[i]):
                agents[i] = new_agent

        new_best_agent = agents[np.argmin([fobj(agent) for agent in agents])]
        new_best_fitness = fobj(new_best_agent)

        if new_best_fitness < best_fitness:
            best_agent = new_best_agent
            best_fitness = new_best_fitness
        Convergence_curve[t] = best_agent
        t = t + 1
    best_agent = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_agent, Convergence_curve, best_fitness, ct
