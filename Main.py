import os
import cv2 as cv
import pandas as pd
from numpy import matlib
import random as rn
from COA import COA
from DO import DO
from Global_Vars import Global_Vars
from Model_Auto import Model_AutoEn_Feat
from Model_Bayesian import Model_Bayesian
from Model_CapsNet import Model_CapsNet
from Model_HDMcCNet import Model_HDMcCNe
from Model_RNN import Model_RNN
from Model_SVM import Model_SVM
from Plot_Results import *
from Proposed import Proposed
from TOA import TOA
from TSO import TSO
from objfun_feat import objfun_feat


def Read_Dataset_1(out_dir):
    Image = []
    Target = []
    in_dir = os.listdir(out_dir)
    for i in range(len(in_dir)):
        folder = out_dir + '/' + in_dir[i]
        file = os.listdir(folder)
        for j in range(len(file)):
            filename = folder + '/' + file[j]
            data = cv.imread(filename)
            height = 512
            weight = 512
            dim = (height, weight)
            resize_iamge = cv.resize(data, dim)
            fold_sp = folder.split('/')
            file_sp = fold_sp[3].split('.')
            Image.append(resize_iamge)
            Target.append(int(file_sp[0]))

    return Image, Target


# Read Dataset
an = 0
if an == 1:
    path = './Dataset/1000images'
    Image, Target = Read_Dataset_1(path)
    np.save('Image.npy', Image)

    # unique coden
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1
    np.save('Target.npy', target)

# Feature Extraction AutoEncoder
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Feat = Model_AutoEn_Feat(Image, Tar)
    np.save('Auto_Feat.npy', Feat)

# Optimization for Weighted Feature
an = 0
if an == 1:
    data = np.load('Auto_Feat.npy', allow_pickle=True)  # Load the Dataset
    Tar = np.load('Target.npy', allow_pickle=True)  # Load the Target
    Best_sol = []
    Target = Tar
    Global_Vars.Data = data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = data.shape[1]
    xmin = matlib.repmat((0.01 * np.ones((1, Chlen))), Npop, 1)
    xmax = matlib.repmat((0.99 * np.ones((1, Chlen))), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = objfun_feat
    max_iter = 50

    print('TSO....')
    [bestfit1, fitness1, bestsol1, Time1] = TSO(initsol, fname, xmin, xmax, max_iter)  # TSO

    print('CO....')
    [bestfit2, fitness2, bestsol2, Time2] = COA(initsol, fname, xmin, xmax, max_iter)  # CO

    print('DO....')
    [bestfit3, fitness3, bestsol3, Time3] = DO(initsol, fname, xmin, xmax, max_iter)  # DO

    print('ESO....')
    [bestfit4, fitness4, bestsol4, Time4] = TOA(initsol, fname, xmin, xmax, max_iter)  # TOA

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BEST_Sol.npy', BestSol)

# Weighted Feature Selection
an = 0
if an == 1:
    Feat = np.load('Auto_Feat.npy', allow_pickle=True)
    bests = np.load('BEST_Sol.npy', allow_pickle=True).astype(int)
    sol = np.round(bests[4, :]).astype(np.int16)
    feat = Feat * sol
    np.save('Weighted_Feature.npy', feat)

# Classification
an = 0
if an == 1:
    Feat = np.load('Weighted_Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    K = 5
    Per = 1 / 5
    Perc = round(Feat.shape[0] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((5, 14))
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        Eval[0, :], pred = Model_SVM(Train_Data, Train_Target, Test_Data,
                                           Test_Target)  # Model SVM
        Eval[1, :], pred1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RNN
        Eval[2, :], pred2 = Model_CapsNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model CapsNet
        Eval[3, :], pred3 = Model_Bayesian(Train_Data, Train_Target, Test_Data, Test_Target)  # Model Bayesian Learning
        Eval[4, :], pred4 = Model_HDMcCNe(Feat, Target)  # Capsnet  + Bayesian
        eval.append(Eval)
    np.save('Eval.npy', eval)  # Save Eval

plotConvResults()
plot_results()
Plot_ROC_Curve()
