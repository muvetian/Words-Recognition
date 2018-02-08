
"""
Sample client code for read_mnist.py.

Author: RR
"""

from read_mnist import load_data, pretty_print
from sklearn.linear_model import SGDClassifier
import pandas as pd
FEATURE = 0
LABEL = 1

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
    print "\nFeatures of third training example:\n"
    pretty_print(train_set[FEATURE][40010])



    clf = SGDClassifier(loss = "hinge", penalty = "l2")
    clf.fit(train_set[FEATURE][0:50000],train_set[LABEL][0:50000])
    counter = 0
    for i in range(100):
        print "the prediction is:" + str(clf.predict(train_set[FEATURE][50000+i]))
        print "the actual one is:" + str(train_set[LABEL][50000+i])
        if(train_set[LABEL][50000+i] == clf.predict(train_set[FEATURE][50000+i])):
            counter+=1
    print counter



    # And here's the label that goes with that training example
    print "\nLabel of first training example:", train_set[LABEL][2], "\n"


    # The test_set is organized in the same way, but only contains 10k
    # examples. Don't touch this data until your model is frozen! Perform all
    # cross-validation, model selection, hyperparameter tuning etc. on the 60k
    # training set. Use the test set simply for reporting performance.


if __name__ == "__main__":
    main()