import math
import random
import numpy as np


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = tnr / fnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)
    fp = sum(Fp)
    tn = sum(Tn)
    fn = sum(Fn)

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    sensitivity = (tp / (tp + fn)) * 100
    specificity = (tn / (tn + fp)) * 100
    precision = (tp / (tp + fp)) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (tn + fp)) * 100
    For = (fn / (fn + tn)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pt = np.math.sqrt(FPR) / (np.math.sqrt(sensitivity) + np.math.sqrt(sensitivity))
    ba = (sensitivity + specificity) / 2
    fm = np.math.sqrt(sensitivity * precision)
    bm = sensitivity + specificity - 100
    mk = precision + NPV - 100
    PLHR = sensitivity / FPR
    lrminus = NegativeLivelihoodRatio(specificity, FNR)
    # dor1 = PLHR / lrminus
    # dor = (tp * tn) / (fp * fn)
    dor2 = (tp * tn) / (fp * tn)
    prevalence = ((tp + fn) / (tp + fp + tn)) * 100
    TS = (tp / (tp + fn + fp)) * 100
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, For, pt,
            ba, fm, bm, mk, PLHR, lrminus, dor2, prevalence, TS]
    # EVAL = [tp, tn, fp, fn, dor, dor1, dor2, precision, PLHR, prevalence]
    return EVAL
