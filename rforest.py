#================================================================================
#
# Class for random forest training
# Another version of what was written by Bruno Rousseau and Me
# The validation of the model is now done with scikit learn module cross_validation
#
#================================================================================
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

class RandomForestModel(object):
    """
    A class that will contain and train data for random forest
    """
    def __init__(self, train_data_fname):
        # Turn data in pandas dataframe
        self.df_train = pd.read_csv(train_data_fname)

    def clean_data(self, df):
        """
        Cleans self.df_train
        """
        ## Sex
        ## female = 0, male = 1
        gender_dict = {'female': 0, 'male': 1}
        #self.df_train.loc[:,'Gender'] = self.df_train.loc[:,'Sex'].map(gender_dict).astype(int)
        df.loc[:,'Gender'] = df.loc[:,'Sex'].map(gender_dict).astype(int)

        ## Port
        # Embarked (at port) from 'C', 'Q', 'S'
        # Could be improved (absolute number do not have real meaning here)
        # Replace NA with most frequent value
        # DataFrame.mode() returns the most frequent object in a set
        # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
        if len(df.Embarked[df.Embarked.isnull() ]) > 0:
            most_common_value = df.Embarked.dropna().mode().values[0]
            df.loc[df.Embarked.isnull(),'Embarked'] = most_common_value 

        # The following lines produce [(0, 'C'), (1, 'Q'), (2, 'S')]
        #Ports = list(enumerate(np.unique(df['Embarked'])))
        # Create dic {port(char): port(int)}
        #Ports_dict = { name : i for i, name in Ports }
        Ports_dict = {'Q': 1, 'C': 0, 'S': 2}
        # Converting port string as port int
        df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

        ## Age
        ##Need to be updated by the class/gender medians
        #median_age = self.df_train['Age'].dropna().median()
        #if len(self.df_train.Age[ self.df_train.Age.isnull() ]) > 0:
        #    self.df_train.loc[ (self.df_train.Age.isnull()), 'Age'] = median_age
        median_ages = np.zeros((2,3))
        ## Get the median ages
        for i in range(0,2):
            for j in range(0, 3):
                median_ages[i, j] = df[(df['Gender']==i) & (df['Pclass']==j+1)]['Age'].dropna().median()
        ## Fill the median age to na
        for i in range(0,2):
            for j in range(0,3):
                df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Age'] = median_ages[i, j]

        ## Fare
        # All the missing Fares -> assume median of their respective class
        if len(df.Fare[ df.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):
                median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):
                df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

        ## Ticket
        ## Tickets are like A/5 21171, let's use the last part as a number
        ticket_number_list=[]
        for iticket in df.Ticket:
            if iticket.split()[-1].isdigit():
                ticket_number_list.append(int( iticket.split()[-1]))
            else:
                ticket_number_list.append(-100000)#Maybe NA would be a better default
        #df['Ticket_number'] = ticket_number_list
        df.loc[:,'Ticket_number'] = ticket_number_list

        ## Pclass
        ## Transforming the class as single (binary)
        dfclass = pd.get_dummies(df['Pclass'], prefix='cl')
        df["cl_1"] = dfclass["cl_1"]
        df["cl_2"] = dfclass["cl_2"]
        df["cl_3"] = dfclass["cl_3"]

    def trainNselfCheck(self, train_fraction = 1):
        """
        Train the model on a fraction of the data and check on another fraction
        Warning: this will affect self.df_train because it calls self.clean_data(self.df_train)
        """

        # Separate the training sample into two samples
        # One for training and one to study the error
        nall = len(self.df_train)
        ntrain = int(train_fraction*nall)

        #train_df = self.df_train.loc[:ntrain-1, :]
        #self.df_train = self.df_train.drop(df.index[-ntrain:])

        ## Training set
        ## Data clean up for training
        self.clean_data(self.df_train)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass'], axis=1)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked'], axis=1)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Age', 'Fare', 'Ticket_number', 'Gender'], axis=1)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked', 'Age', 'Fare', 'Pclass'], axis=1)

        ## Convert to numpy array
        train_data = self.df_train.values
        train_data = train_data[:ntrain,:]

        ## Training
        print 'training with %d examples...'%len(train_data)
        forest = RandomForestClassifier(n_estimators=40)
        forest = forest.fit( train_data[0:,1:], train_data[0:,0] )


        ## Determine the score based of multiple cross validation
        scores = cross_validation.cross_val_score(forest, train_data[0:,1:], train_data[0:,0], cv=25)

        #predictions = forest.predict(validation_data[0::,1::]).astype(int)
        print '\nScores'
        print scores
        print "\n\nCross_validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

        print '\nFeatures importances'
        for iimpotance, ifeature in zip(forest.feature_importances_, list(self.df_train.columns.values)[1:]):
            print '{0} \t: {1} '.format(ifeature, round(iimpotance, 2))
        #raw_input('ok...')
        return {'scores': scores}

    def validation_curves(self):
        """
        Based on the scikit-learn documentation:
        http://scikit-learn.org/stable/modules/learning_curve.html
        possible scoring for randomforest:
        ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        """
        ## Data clean up for training
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked', 'Pclass'], axis=1)
        train_data = self.df_train.values

        ##Validation curves
        paramater4validation = "n_estimators"
        param_range = [1,2,4,8,16,32,64,100]
        train_scores, test_scores = validation_curve(
            RandomForestClassifier(), train_data[0:,1:], train_data[0:,0], param_name=paramater4validation, param_range=param_range,
            cv=10, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.title("Validation Curve")
        plt.xlabel(paramater4validation)
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        plt.plot(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
        plt.show()


    def learning_curves(self, score='accuracy', nestimators=40):
        """
        Creates a plot score vs # of training examples
        possible score:
        ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        more info here:
        http://scikit-learn.org/stable/modules/learning_curve.html
        """
        ## Data clean up for training
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked', 'Pclass'], axis=1)
        train_data = self.df_train.values
        X = train_data[0:,1:]
        y = train_data[0:,0]
        train_sizes = [x / 10.0 for x in range(1, 11)]##Can be other formats

        print 'learning...'
        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=nestimators), X, y, cv=10, n_jobs=1, train_sizes=train_sizes, scoring=score)

        ## Plotting
        plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel(score)
        plt.title("Learning Curves (RandomForest n_estimators={0})".format(nestimators))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
        plt.legend(loc="best")
        print 'Done'
        plt.show()

    def make_prediction(self, test_data_fname):
        """
        Predict the survival on the on the csv
        """
        df_test = pd.read_csv(test_data_fname)
        
        #######
        ## Cleaning

        ## Training set
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked'], axis=1)
        train_data = self.df_train.values
        ## Test set
        self.clean_data(df_test)
        ids = df_test['PassengerId'].values
        df_test = df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked'], axis=1)
        test_data = df_test.values

        ##Training
        print 'training...'
        forest = RandomForestClassifier(n_estimators=40)
        forest = forest.fit( train_data[0:,1:], train_data[0:,0])

        print '\nFeatures importances'
        for iimpotance, ifeature in zip(forest.feature_importances_, list(self.df_train.columns.values)[1:]):
            print '{0} \t: {1} '.format(ifeature, round(iimpotance, 2))

        
        ##Predicting
        print 'predicting...'
        predictions = forest.predict(test_data).astype(int)
        
        # Writing output
        predictions_file = open("submitforest.csv", "wb")
        open_file_object = csv.writer(predictions_file)    
        open_file_object.writerow(["PassengerId","Survived"])
        open_file_object.writerows(zip(ids, predictions))
        predictions_file.close()
        print 'Done.'

        

if __name__=='__main__':
    rfmodel = RandomForestModel('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/train.csv')
    rfmodel.trainNselfCheck()
    #rfmodel.make_prediction('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/test.csv')
    #rfmodel.validation_curves()
    #rfmodel.learning_curves()
