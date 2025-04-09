import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO', 'CO', 'DO', 'TOA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[0, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(50)
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='TSO')
    plt.plot(length, Conv_Graph[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
             markersize=12, label='CO')
    plt.plot(length, Conv_Graph[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='DOA')
    plt.plot(length, Conv_Graph[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='TOA')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='RF-TOA')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    # plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['SVM', 'RNN', 'CapsNet', 'Bayesian Learning', 'HDMcCNet']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')

    colors = cycle(["blue", "darkorange", "y", "deeppink", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    # path = "./Results/ROC.png"
    # plt.savefig(path)
    plt.show()


def plot_results():
    eval = np.load('Eval.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC', 'FOR', 'PT',
            'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20]
    Algorithm = ['TERMS', 'TSO', 'CO', 'DO', 'TOA', 'PROPOSED']
    Classifier = ['TERMS', 'SVM', 'RNN', 'CapsNet', 'Bayesian Learning', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 2):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        Table.add_column(Classifier[5], value1[4, :])
        print('-------------------------------------------------- KFOLD - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#4EEE94', width=0.10, label="SVM")
            ax.bar(X + 0.10, Graph[:, 6], color='#9A32CD', width=0.10, label="RNN")
            ax.bar(X + 0.20, Graph[:, 7], color='#FF1493', width=0.10, label="CapsNet")
            ax.bar(X + 0.30, Graph[:, 8], color='#FFC125', width=0.10, label="Bayesian Learning")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="HDMcCNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFold')
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_results()
    Plot_ROC_Curve()
