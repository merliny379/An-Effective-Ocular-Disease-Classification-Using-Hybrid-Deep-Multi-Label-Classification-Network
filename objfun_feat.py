import numpy as np
from Global_Vars import Global_Vars
from Relief_score import reliefF


def objfun_feat(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            data1 = data * sol
            rscore = reliefF(np.array(data1), np.array(Tar.reshape(-1)))
            Fitn[i] = 1 / rscore
        return Fitn
    else:
        sol = Soln
        data1 = data * sol
        rscore = reliefF(np.array(data1), np.array(Tar.reshape(-1)))
        Fitn = 1 / rscore
        return Fitn
