from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np

from Evaluation import evaluation


def Model_SVM(train_data, train_target, test_data, test_target, sol=0):
    sol = int(sol)
    Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    if sol == 1:
        clf = SVC(kernel=Kernels[sol], degree=8)
    else:
        clf = SVC(kernel=Kernels[sol])

    pred = np.zeros(test_target.shape)
    # fitting x samples and y classes
    for i in range(test_target.shape[1]):
        clf.fit(train_data.tolist(), train_target[:, i].tolist())

        Y_pred = clf.predict(test_data.tolist())
        pred[:, i] = np.asarray(Y_pred)

    Eval = evaluation(pred, test_target)
    return Eval, pred