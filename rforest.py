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
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class RandomForestModel(object):
    """
    A class that will contain and train data for random forest
    """
    def __init__(self, train_data_fname):
        # Turn data in pandas dataframe
        self.df_train = pd.read_csv(train_data_fname)

    def clean_data(self):
        """
        Cleans self.df_train
        """
        ## Sex
        ## female = 0, male = 1
        gender_dict = {'female': 0, 'male': 1}
        self.df_train.loc[:,'Gender'] = self.df_train.loc[:,'Sex'].map(gender_dict).astype(int)

        ## Port
        # Embarked (at port) from 'C', 'Q', 'S'
        # Could be improved (absolute number do not have real meaning here)
        # Replace NA with most frequent value
        # DataFrame.mode() returns the most frequent object in a set
        # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
        if len(self.df_train.Embarked[self.df_train.Embarked.isnull() ]) > 0:
            most_common_value = self.df_train.Embarked.dropna().mode().values[0]
            self.df_train.loc[self.df_train.Embarked.isnull(),'Embarked'] = most_common_value 

        # The following lines produce [(0, 'C'), (1, 'Q'), (2, 'S')]
        #Ports = list(enumerate(np.unique(df['Embarked'])))
        # Create dic {port(char): port(int)}
        #Ports_dict = { name : i for i, name in Ports }
        Ports_dict = {'Q': 1, 'C': 0, 'S': 2}
        # Converting port string as port int
        self.df_train.Embarked = self.df_train.Embarked.map( lambda x: Ports_dict[x]).astype(int)

        ## Age
        ##Need to be updated by the class/gender medians
        median_age = self.df_train['Age'].dropna().median()
        if len(self.df_train.Age[ self.df_train.Age.isnull() ]) > 0:
            self.df_train.loc[ (self.df_train.Age.isnull()), 'Age'] = median_age


        ## Fare
        # All the missing Fares -> assume median of their respective class
        if len(self.df_train.Fare[ self.df_train.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):
                median_fare[f] = self.df_train[ self.df_train.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):
                self.df_train.loc[ (self.df_train.Fare.isnull()) & (self.df_train.Pclass == f+1 ), 'Fare'] = median_fare[f]

        ## Ticket
        ## Tickets are like A/5 21171, let's use the last part as a number
        ticket_number_list=[]
        for iticket in self.df_train.Ticket:
            if iticket.split()[-1].isdigit():
                ticket_number_list.append(int( iticket.split()[-1]))
            else:
                ticket_number_list.append(-100000)#Maybe NA would be a better default
        #df['Ticket_number'] = ticket_number_list
        self.df_train.loc[:,'Ticket_number'] = ticket_number_list

    def trainNselfCheck(self, train_fraction = 1):
        """
        Train the model on a fraction of the data and check on another fraction
        Warning: this cleans the data frame self.df_train
        """

        # Separate the training sample into two samples
        # One for training and one to study the error
        nall = len(self.df_train)
        ntrain = int(train_fraction*nall)

        #train_df = self.df_train.loc[:ntrain-1, :]
        #self.df_train = self.df_train.drop(df.index[-ntrain:])

        ## Training set
        ## Data clean up for training
        self.clean_data()
        self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        ## Convert to numpy array
        train_data = self.df_train.values
        train_data = train_data[:ntrain,:]

        ## Training
        print 'training with %d examples...'%len(train_data)
        forest = RandomForestClassifier(n_estimators=40)
        forest = forest.fit( train_data[0:,1:], train_data[0:,0] )


        ## Determine the score based of multiple cross validation
        scores = cross_validation.cross_val_score(forest, train_data[0:,1:], train_data[0:,0], cv=5)

        #predictions = forest.predict(validation_data[0::,1::]).astype(int)
        print '\nScores'
        print scores
        print "\n\nCross_validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #raw_input('ok...')
        return {'scores': scores}



if __name__=='__main__':
    rfmodel = RandomForestModel('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/train.csv')
    rfmodel.trainNselfCheck()
