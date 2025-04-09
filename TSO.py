import math
import random
import time
import numpy as np


# Tuna Swarm Optimization Algorithm

def TSO(Positions, fobj, lb, ub, Max_iter):
    N, Dim = Positions.shape[0], Positions.shape[1]
    Tuna1 = np.zeros((1, Dim))
    Tuna1_fit = float('inf')

    Iter = 0
    aa = 0.7
    z = 0.05
    t = 0
    ct = time.time()

    Convergence_curve = np.zeros((Max_iter, 1))

    while Iter < Max_iter:
        C = Iter / Max_iter
        a1 = aa + (1 - aa) * C
        a2 = (1 - aa) - (1 - aa) * C
        for i in range(Positions.shape[0]):

            Flag4ub = Positions[i, :] > ub[i, :]
            Flag4lb = Positions[i, :] < lb[i, :]
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub[i, :] * Flag4ub + lb[i, :] * Flag4lb

            fitness = fobj(Positions[i, :])

            if fitness < Tuna1_fit:
                Tuna1_fit = fitness
                Tuna1 = Positions[i, :]

        t = (1 - Iter / Max_iter) ** (Iter / Max_iter)

        if random.random() < z:
            Positions[1, :] = (ub[1, :] - lb[1, :]) * (random.random + lb[1, :])
        else:
            if 0.5 < random.random():
                r1 = random.random()
                Beta = math.exp(r1 * math.exp(3 * math.cos(math.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                    math.cos(2 * math.pi * r1))
                if C > random.random():
                    Positions[1, :] = a1 * (Tuna1 + Beta * abs(Tuna1 - Positions[1, :])) + a2 * Positions[1, :]  # Equation(8.3)
                else:
                    IndivRand = random.uniform(1, Dim) * (ub[1, :] - lb[1, :]) + lb[1, :]
                    Positions[1, :] = a1 * (IndivRand + Beta * abs(IndivRand - Positions[i, :])) + a2 * Positions[1, :]  # Equation(8.1)
            else:
                TF = (random.random() > 0.5) * 2 - 1
                if 0.5 > random.random():
                    Positions[1, :] = Tuna1 + random.uniform(Dim, 1) * (Tuna1 - Positions[1, :]) + TF * t ** 2. * (
                                Tuna1 - Positions[1, :])  # Equation(9.1)
                else:
                    Positions[1, :] = TF * t ** 2. * Positions[1, :]  # Equation(9.2)
        for i in range(1, N):
            if random.random() < z:
                Positions[i, :] = (ub[i, :] - lb[i, :]) * random.random() + lb[i, :]
            else:
                if 0.5 < random.random():
                    r1 = random.random()
                    Beta = math.exp(r1 * math.exp(3 * math.cos(math.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                        math.cos(2 * math.pi * r1))
                    if C > random.random():
                        Positions[i, :] = a1 * (Tuna1 + Beta * abs(Tuna1 - Positions[i, :])) + a2 * Positions[i - 1, :]  # Equation(8.4)
                    else:
                        IndivRand = random.uniform(1, Dim) * (ub[i, :] - lb[i, :]) + lb[i, :]
                        Positions[i, :] = a1 * (IndivRand + Beta * abs(IndivRand - Positions[i, :])) + a2 * Positions[i - 1, :]  # Equation(8.2)
                else:
                    TF = (random.random() > 0.5) * 2 - 1
                    if 0.5 > random.random():
                        Positions[i, :] = Tuna1 + random.uniform(1,Dim) * (Tuna1 - Positions[i, :]) + TF * t ** 2. * (Tuna1 - Positions[i, :])  # Equation(9.1)
                    else:
                        Positions[i, :] = TF * t ** 2. * Positions[i, :]  # Equation(9.2)

        Convergence_curve[Iter] = Tuna1_fit
        Iter = Iter + 1
    Tuna1_fit = Convergence_curve[Max_iter - 1][0]
    Time = time.time() - ct
    return Tuna1_fit, Convergence_curve, Tuna1, Time
