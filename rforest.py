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
        self.iscleaned = False

    def substrings_in_string(self, big_string, substrings):
        """
        Returns the substrings if found in big_string
        else return np.nan
        """
        for substring in substrings:
            if substring in big_string:
                return substring
        print big_string
        return np.nan
    def replace_titles(self, x):
        title=x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 0#Mr
        elif title in ['Countess', 'Mme']:
            return 1#Mrs
        elif title in ['Mlle', 'Ms']:
            return 2#Miss
        elif title =='Dr':
            if x['Sex']=='Male':
                return 0#Mr
            else:
                return 1#Mrs
        else:
            #return title
            return 3


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
        dfembk = pd.get_dummies(df['Embarked'], prefix='port')
        df["port_0"] = dfembk["port_0"]
        df["port_1"] = dfembk["port_1"]
        df["port_2"] = dfembk["port_2"]

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
        df['Age'] = (df['Age'] - df['Age'].mean()) / (df['Age'].max() - df['Age'].min())

        ## Fare
        # All the missing Fares -> assume median of their respective class
        if len(df.Fare[ df.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):
                median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):
                df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
        df['Fare'] = (df['Fare'] - df['Fare'].mean()) / (df['Fare'].max() - df['Fare'].min())

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
        # Nomalize the ticket number (there must be something to do this in scikit learn)
        df['Ticket_number'] = (df['Ticket_number'] - df['Ticket_number'].mean()) / (df['Ticket_number'].max() - df['Ticket_number'].min())

        ## Pclass
        ## Transforming the class as single (binary)
        dfclass = pd.get_dummies(df['Pclass'], prefix='cl')
        df["cl_1"] = dfclass["cl_1"]
        df["cl_2"] = dfclass["cl_2"]
        df["cl_3"] = dfclass["cl_3"]

        ## Title (from name)
        ## As suggested here:
        ## http://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
        title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
        df['Title'] = df['Name'].map(lambda x: self.substrings_in_string(x, title_list))
        df['Title']=df.apply(self.replace_titles, axis=1)
        dftitle = pd.get_dummies(df['Title'], prefix='title')
        
        df["title_0"] = dftitle["title_0"]
        df["title_1"] = dftitle["title_1"]
        df["title_2"] = dftitle["title_2"]
        df["title_3"] = dftitle["title_3"]

        df.drop(['Title',], axis=1, inplace=True)
        self.iscleaned = True
        return df

    def trainNselfCheck(self, train_fraction = 1):
        """
        Train the model on a fraction of the data and check on another fraction
        Warning: this will affect self.df_train because it calls self.clean_data(self.df_train)
        """
        columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass']
        #columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Parch', 'Embarked', 'cl_1', 'cl_2', 'cl_3','title_1','title_2']
        # Separate the training sample into two samples
        # One for training and one to study the error
        nall = len(self.df_train)
        ntrain = int(train_fraction*nall)

        #train_df = self.df_train.loc[:ntrain-1, :]
        #self.df_train = self.df_train.drop(df.index[-ntrain:])

        ## Training set
        ## Data clean up for training
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(columns2drop, axis=1)

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
        #columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked', 'Pclass']
        columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Parch', 'Embarked', 'cl_1', 'cl_2', 'cl_3','title_1','title_2','Gender']

        ## Data clean up for training
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(columns2drop, axis=1)
        print 'Training on the following features:'
        print list(self.df_train.columns.values)
        train_data = self.df_train.values
        X = train_data[0:,1:]
        y = train_data[0:,0]
        train_sizes = [x / 10.0 for x in range(1, 11)]##Can be other formats

        print 'learning...'
        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=nestimators), X, y, cv=10, n_jobs=1, train_sizes=train_sizes, scoring=score)

        ## Plotting
        fig = plt.figure()
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
        fig.show()
        raw_input('press enter when finished...')

    def make_prediction(self, test_data_fname):
        """
        Predict the survival on the on the csv
        """
        df_test = pd.read_csv(test_data_fname)
        #columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp', 'Embarked', 'Pclass']
        columns2drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass', 'Parch', 'Embarked', 'cl_1', 'cl_2', 'cl_3','title_1','title_2','Gender','SibSp']

        #######
        ## Cleaning

        ## Training set
        self.clean_data(self.df_train)
        self.df_train = self.df_train.drop(columns2drop, axis=1)
        train_data = self.df_train.values
        ## Test set
        self.clean_data(df_test)
        ids = df_test['PassengerId'].values
        df_test = df_test.drop(columns2drop, axis=1)
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

    def show_feature(self, feature):
        """
        Plot the given feature (after cleaning)
        """
        self.clean_data(self.df_train)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        #self.df_train = self.df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Pclass'], axis=1)
        nbins = 50##Maybe a dic with this format {feature: nbins}
        barwidth=1/nbins
        #nbins = 1000
        fig = plt.figure(figsize=(8,10))
        #plt.title(feature) ##Don't know why this messes up the axis

        ## Absolute distribution
        ax = fig.add_subplot(3,1,1)
        ax.hist([self.df_train[self.df_train.Survived==1][feature].values, self.df_train[self.df_train.Survived==0][feature].values], alpha=0.5, bins=nbins, stacked=True, label=['Survived', 'Died'], color=['red', 'blue'], edgecolor='white', width=barwidth)
        plt.grid()
        ax.set_ylabel('# of passengers')

        ##Relative survival distribution
        featureCuts, cutbins = pd.cut(self.df_train[feature], nbins, retbins=True)
        grouped_survived = self.df_train[self.df_train.Survived==1]['Fare'].groupby(featureCuts)
        grouped_died = self.df_train[self.df_train.Survived==0]['Fare'].groupby(featureCuts)
        died_fracs = []
        survived_fracs = []
        filled_bins = []
        SurvivedOverDied_list = []
        SurvivedOverDied_filled_bins = []
        for idied, isurvived, ibin in zip(grouped_died.count(), grouped_survived.count(), cutbins):
            isum = idied + isurvived
            if isum <1:
                continue
            died_fracs.append(idied/(idied + isurvived))
            survived_fracs.append(isurvived/(idied + isurvived))
            filled_bins.append(ibin)
            if idied >=1:
                SurvivedOverDied_list.append(isurvived/idied)
                SurvivedOverDied_filled_bins.append(ibin)
        ax2 = fig.add_subplot(3,1,2)
        plt.bar(filled_bins, survived_fracs, color='red', edgecolor='white',alpha=0.5, width=barwidth, label='Survived')
        plt.bar(SurvivedOverDied_filled_bins, SurvivedOverDied_list, color='blue', edgecolor='white',alpha=0.5, width=barwidth, label='Died')
        ax2.set_ylabel('Fraction')
        plt.legend(loc='best')
        plt.grid()

        ax3 = fig.add_subplot(3,1,3)
        plt.bar(filled_bins, survived_fracs, color='green', edgecolor='white',alpha=0.5, width=barwidth)
        ax3.set_ylabel('Survived/Died')
        ax3.set_xlabel(feature)
        plt.subplots_adjust(hspace=0)
        plt.grid()
        fig.show()
        raw_input('press enter when finished')

        

if __name__=='__main__':
    rfmodel = RandomForestModel('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/train.csv')
    rfmodel.trainNselfCheck()
    #rfmodel.make_prediction('/Users/jean-francoisrajotte/projects/kaggle/titanic/data/test.csv')
    #rfmodel.validation_curves()
    #rfmodel.learning_curves()
    #rfmodel.show_feature('Ticket_number')
    #rfmodel.show_feature('Fare')
