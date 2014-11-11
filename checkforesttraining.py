from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from myfirstforest import clean_data


def checktraining(full_df, train_fraction = 0.7):
    """Trains a fraction of the data frame sample
    and checks the performance on the rest of the sample
    Default fraction for training is as suggested by Andrew Ng in his ML course
    """

    # Separate the training sample into two sample
    # One for training and one to study the error
    ntrain = int(train_fraction*len(full_df))

    train_df = full_df[0:ntrain]
    check_df = full_df[ntrain:]

    ##Clean the two samples

    ##Training sample
    train_df = clean_data(train_df)
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    ##Checking sample
    check_df = clean_data(check_df)
    check_df = check_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    ###### Convert data frame to numpy array
    train_data = train_df.values
    check_data = check_df.values


    print 'Training...'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

    print 'Predicting...'
    output = forest.predict(check_data[0::,1::]).astype(int)

    print 'Comparing...'

    ##Prediction result variables
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    ##Loop over all passagers and fill the prediction results variables
    for isurvived in zip(output, check_data[0::,0].astype(int)):
        ipredicted = isurvived[0]
        itrue = isurvived[1]
        if ipredicted:#Prediction: passager survived
            if itrue:#Passager did survive
                true_positive += 1
            else:
                false_positive += 1
        else:#Prediction: passager died
            if itrue:#Passager did survive
                false_positive += 1
            else:
                true_negative += 1

    ##Now compute some stats
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1score = 2*precision*recall/(precision + recall)


    print '\n\n\n' + 20*'---' + '\nTraining performance summary:\n'
    print 'True positive\t{0}'.format(true_positive)
    print 'True negative\t{0}'.format(true_negative)
    print 'False positive\t{0}'.format(false_positive)
    print 'False negative\t{0}'.format(false_negative)

    print '\nAccuracy:\t{0}'.format(round(accuracy, 2))
    print 'Precision:\t{0}'.format(round(precision, 2))
    print 'Recall: \t{0}'.format(round(recall, 2))
    print 'F1Score:\t{0}'.format(round(f1score,2))

    print '\n'

if __name__=='__main__':
    # Data file name
    traincsvname = 'data/train.csv'
    fullsample_df = pd.read_csv(traincsvname, header=0)
    training_fraction = 0.7
    checktraining(fullsample_df, training_fraction)
