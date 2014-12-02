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

    def clean_data(self, df):
        """
        Returns a data frame formatted for model training
        """
        ## Sex
        ## female = 0, male = 1
        df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

        ## Port
        # Embarked (at port) from 'C', 'Q', 'S'
        # Could be improved (absolute number do not have real meaning here)
        # Replace NA with most frequent value
        # DataFRame.mode() returns the most frequent object in a set
        # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
        if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
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
        median_age = df['Age'].dropna().median()
        if len(df.Age[ df.Age.isnull() ]) > 0:
            df.loc[ (df.Age.isnull()), 'Age'] = median_age


        ## Fare
        # All the missing Fares -> assume median of their respective class
        if len(df.Fare[ df.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):
                median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):
                df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

        return df

    def trainNselfCheck(self, train_fraction = 1):
        """
        Train the model on a fraction of the data and check on another fraction
        """

        # Separate the training sample into two samples
        # One for training and one to study the error
        nall = len(self.df_train)
        ntrain = int(train_fraction*nall)

        train_df = self.df_train[0:ntrain]

        ## Training set
        ## Data clean up for training
        train_df = self.clean_data(train_df)
        train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        ## Convert to numpy array
        train_data = train_df.values

        ## Training
        print 'training with %d examples...'%len(train_data)
        forest = RandomForestClassifier(n_estimators=40)
        forest = forest.fit( train_data[0:,1:], train_data[0:,0] )


        ## Determine the score based of multiple cross validation
        scores = cross_validation.cross_val_score(forest, train_data[0:,1:], train_data[0:,0], cv=10)

        #predictions = forest.predict(validation_data[0::,1::]).astype(int)
        print 'Scores'
        print scores
        print "Cross_validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #raw_input('ok...')


if __name__=='__main__':
    rfmodel = RandomForestModel('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/train.csv')
    rfmodel.trainNselfCheck()
