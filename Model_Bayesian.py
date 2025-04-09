from sklearn.naive_bayes import GaussianNB
from Evaluation import evaluation


def Model_Bayesian(train_data, train_target, test_data, test_target):
    # Initialize the Gaussian Naive Bayes classifier
    classifier = GaussianNB()
    # Train the classifier
    classifier.fit(train_data, train_target)
    # Predict on the test set
    pred = classifier.predict(test_target)
    Eval = evaluation(pred, test_target)
    return Eval, pred
