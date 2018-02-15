
"""
Sample client code for read_mnist.py.

Author: RR
"""

from read_mnist import load_data, pretty_print
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
FEATURE = 0
LABEL = 1
def feature_selection_chi2(train_set,num):
    new_train_set = SelectKBest(chi2, k=num).fit_transform(train_set[FEATURE],
    train_set[LABEL])
    print "new feature number is" + str(new_train_set.shape)
    return new_train_set
def feature_selection_variance(train_set,threshold):
    selector = VarianceThreshold()
    filtered = selector.fit_transform(train_set)
    return filtered

def MLPclassifier(clf,train_set,test_set,train_number,test_number):
    clf.fit(train_set[FEATURE][0:train_number],train_set[LABEL][0:train_number])
    counter = 0
    y_true = []
    Y_pred = []

    for i in range(test_number):
        true = test_set[LABEL][i]
        predict = clf.predict(test_set[FEATURE][i])
        y_true.append(true)
        y_pred.append(predict)
        if(true == predict):
            counter+=1
    macro = precision_score(y_true, y_pred, average='macro')
    micro = precision_score(y_true, y_pred, average='micro')
    weighted = precision_score(y_true, y_pred, average='weighted')

    return (counter, weighted)
def SGDclassifier(clf,train_set,test_set,train_number,test_number):
    clf.fit(train_set[FEATURE][0:train_number],train_set[LABEL][0:train_number])
    counter = 0
    y_true = []
    Y_pred = []

    for i in range(test_number):
        true = test_set[LABEL][i]
        predict = clf.predict(test_set[FEATURE][i])
        y_true.append(true)
        y_pred.append(predict)
        if(true == predict):
            counter+=1
    macro = precision_score(y_true, y_pred, average='macro')
    micro = precision_score(y_true, y_pred, average='micro')
    weighted = precision_score(y_true, y_pred, average='weighted')

    return (counter, macro, micro, weighted)
def main():
    """ Example of how to load and parse MNIST data. """

    train_set, test_set = load_data()

    # train_set is a two-element tuple. The first element, i.e.,
    # train_set[0] is a 60,000 x 784 numpy matrix. There are 60k
    # rows in the matrix, each row corresponding to a single example.
    # There are 784 columns, each corresponding to the value of a
    # single pixel in the 28x28 image.
    print "\nDimensions of training set feature matrix:",
    print train_set[FEATURE].shape

    # The labels for each example are maintained separately in train_set[1].
    # This is a 60,000 x 1 numpy matrix, where each element is the label
    # for the corresponding training example.
    print "\nDimensions of training set label matrix:", train_set[LABEL].shape

    # Example of how to access a individual training example (in this case,
    # the third example, i.e., the training example at index 2). We could
    # also just use print to output it to the screen, but pretty_print formats
    # the data in a nicer way: if you squint, you should be able to make out
    # the number 4 in the matrix data.
    # print "\nFeatures of third training example:\n"
    #

    #
    filtered = feature_selection_variance(train_set[FEATURE],0)
    feature_selection_chi2(train_set,700)

    # clf = SGDClassifier(loss = "hinge", penalty = "l2")
    # clf.fit(train_set[FEATURE][0:60000],train_set[LABEL][0:60000])
    #
    #
    # clf = MLPClassifier()
    # clf.fit(train_set[FEATURE][0:60000],train_set[LABEL][0:60000])
    #



    # And here's the label that goes with that training example
    print "\nLabel of first training example:", train_set[LABEL][2], "\n"


    # The test_set is organized in the same way, but only contains 10k
    # examples. Don't touch this data until your model is frozen! Perform all
    # cross-validation, model selection, hyperparameter tuning etc. on the 60k
    # training set. Use the test set simply for reporting performance.


if __name__ == "__main__":
    main()
