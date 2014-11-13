from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from myfirstforest import clean_data



def rftraining_OLD(train_df, check_df, n_estim = 10):
    """Trains (with random forest) a fraction of the data frame sample
    and checks the performance on the rest of the sample
    Default fraction for training is as suggested by Andrew Ng in his ML course
    """
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
    #forest = RandomForestClassifier(n_estimators=100)
    forest = RandomForestClassifier(n_estimators=n_estim)
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


    out_dic = {}
    out_dic['true_positive'] = true_positive
    out_dic['true_negative'] = true_negative
    out_dic['false_positive'] = false_positive
    out_dic['false_negative'] = false_negative
    out_dic['accuracy'] = accuracy
    out_dic['precision'] = precision
    out_dic['recall'] = recall
    out_dic['F1Score'] = f1score
    return out_dic


def rftraining(train_df, n_estim = 40):
    """Trains (with random forest) a fraction of the data frame sample
    and checks the performance on the rest of the sample
    Default fraction for training is as suggested by Andrew Ng in his ML course
    """
    ##Clean
    train_df = clean_data(train_df)
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


    ###### Convert data frame to numpy array
    train_data = train_df.values

    print 'Training...'

    forest = RandomForestClassifier(n_estimators=n_estim)
    return forest.fit( train_data[0::,1::], train_data[0::,0] )
    


def check_training(trained_forest, check_df):
    """Determine some basic validation parameters on the check_df data frame
    trained_forest is the output from RandomForestClassifier fit
    check_df is the data fram to predict the output on
    """

    check_df = clean_data(check_df)
    check_df = check_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    check_data = check_df.values
    
    output = trained_forest.predict(check_data[0::,1::]).astype(int)

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


    out_dic = {}
    out_dic['true_positive'] = true_positive
    out_dic['true_negative'] = true_negative
    out_dic['false_positive'] = false_positive
    out_dic['false_negative'] = false_negative
    out_dic['accuracy'] = accuracy
    out_dic['precision'] = precision
    out_dic['recall'] = recall
    out_dic['F1Score'] = f1score
    return out_dic

    
def rfFractiontraining(full_df, train_fraction = 0.7, n_estim = 40):
    """Trains (with random forest) a fraction of the data frame sample
    and checks the performance on the rest of the sample
    Default fraction for training is as suggested by Andrew Ng in his ML course
    """

    # Separate the training sample into two sample
    # One for training and one to study the error
    ntrain = int(train_fraction*len(full_df))

    train_df = full_df[0:ntrain]
    check_df = full_df[ntrain:]
    trained_forest = rftraining(train_df, n_estim)
    return check_training(trained_forest, check_df)

def show_training_summary(full_df, train_fraction = 0.7, n_estim = 10):
    """Shows the training summary
    """
    train_dic = rfFractiontraining(full_df, train_fraction, n_estim)
    print train_dic
    print '\n\n\n' + 20*'---' + '\nTraining performance summary:\n'
    print 'True positive\t{0}'.format(train_dic['true_positive'])
    print 'True negative\t{0}'.format(train_dic['true_negative'])
    print 'False positive\t{0}'.format(train_dic['false_positive'])
    print 'False negative\t{0}'.format(train_dic['false_negative'])

    print '\nAccuracy:\t{0}'.format(round(train_dic['accuracy'], 2))
    print 'Precision:\t{0}'.format(round(train_dic['precision'], 2))
    print 'Recall: \t{0}'.format(round(train_dic['recall'], 2))
    print 'F1Score:\t{0}'.format(round(train_dic['F1Score'],2))

    print '\n'


def vary_nestim(full_df, train_fraction = 0.7):
    """Shows the results for different estimators
    """
    nestimators_list = range(1, 20) + range(25,100,4)
    all_train = {}#Will contain the results of all nestimator
    f1scores = []
    accuracies = []
    precisions = []
    for inestimators in nestimators_list:
        #print inestimators
        all_train[inestimators] = rfFractiontraining(full_df, train_fraction, inestimators)
        f1scores.append(all_train[inestimators]['F1Score'])
        accuracies.append(all_train[inestimators]['accuracy'])
        precisions.append(all_train[inestimators]['precision'])
    fig1, ax1 = plt.subplots()
    ax1.plot(nestimators_list, f1scores,marker='o', label='F1Score')
    ax1.plot(nestimators_list, accuracies,marker='x', label='Accuracy')
    ax1.plot(nestimators_list, precisions,marker='s', label='Precision')
    ax1.set_xlabel('nestimators')
    #ax1.set_ylim([0.3,0.9])
    ax1.legend(loc='lower right')
    fig1.show()
    raw_input('press enter when finished...')
    
def vary_nsamples(full_df, validation_fraction = 0.3, n_estimators=40):
    """Shows the results as functions example in the training sample
      verification_fraction is the sample to test the result from training
      (verification sample is never in the training)
    """

    # Separate the training sample into two sample
    # One for training and one to study the error
    nvalidation = int(validation_fraction*len(full_df))

    train_df = full_df[0:-nvalidation]
    check_df = full_df[nvalidation:]
    nsample_list = range(50, len(train_df), 50)
    all_train = {}
    f1scores = []
    accuracies = []
    precisions = []
    ##Variables determined on the passangers in the training sample
    f1scores_self = []
    accuracies_self = []
    precisions_self = []
    for insample in nsample_list:
        itrain_df = train_df[:insample]
        icheck_df = full_df[nvalidation:]##Making a copy each time is not the smartest but it gets cleaned in check_training so this function would have to be modified before
        itrained_forest = rftraining(itrain_df, n_estimators)

        ## Check the model on the independant sample
        all_train[insample] = check_training(itrained_forest, icheck_df)
        f1scores.append(all_train[insample]['F1Score'])
        accuracies.append(all_train[insample]['accuracy'])
        precisions.append(all_train[insample]['precision'])

        ##Check the model on the training sample
        ## A copy must be done to because it would change the original
        ifulltrain_df_copy = train_df[0::]
        iself_check = check_training(itrained_forest, ifulltrain_df_copy)
        f1scores_self.append(iself_check['F1Score'])
        accuracies_self.append(iself_check['accuracy'])
        precisions_self.append(iself_check['precision'])
        print insample
    print f1scores
    fig1, ax1 = plt.subplots()
    ax1.plot(nsample_list, f1scores,marker='o', label='F1Score')
    ax1.plot(nsample_list, accuracies,marker='x', label='Accuracy')
    ax1.plot(nsample_list, precisions,marker='s', label='Precision')
    ax1.set_xlabel('# of passagers for training')
    ax1.legend(loc='lower right')
    fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(nsample_list, f1scores,marker='o', label='F1Score (indep)')
    ax2.plot(nsample_list, f1scores_self,marker='o', label='F1Score (self)')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('# of passagers for training')
    fig2.show()

    fig3, ax3 = plt.subplots()
    ax3.plot(nsample_list, precisions,marker='s', label='Precision (indep)')
    ax3.plot(nsample_list, precisions_self,marker='s', label='Precision (self)')
    ax3.set_xlabel('# of passagers for training')
    ax3.legend(loc='lower right')
    fig3.show()

    raw_input('press enter when finished...')

    
if __name__=='__main__':
    # Data file name
    traincsvname = 'data/train.csv'
    fullsample_df = pd.read_csv(traincsvname, header=0)
    training_fraction = 0.7

    #This is to show the text summary of one training
    #nestimators = 50
    #show_training_summary(fullsample_df, training_fraction, nestimators)

    #This is to see the fit results vs n_estimators
    vary_nestim(fullsample_df, training_fraction)

    #This is to see the fit results vs number of passagers
    #vary_nsamples(fullsample_df)
