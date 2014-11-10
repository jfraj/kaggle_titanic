import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier

# Data file name
traincsvname = 'data/train.csv'
testcsvname = 'data/test.csv'


def clean_data(df):
    """Returns a cleaned data frame
    Cleans the dataframe
    Changes some strings into integer

    note: Ports_dict is hard coded so it is the same for any give data frame
    if a new port appears in a new data sample, just add a new key to Ports_dict
    """
    ## Sex: female = 0, male = 1
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    ## Port
    # Embarked (at port) from 'C', 'Q', 'S'
    # Could be improved (absolute number do not have real meaning here)
    # Replace NA with most frequent value
    # DataFRame.mode() returns the most frequent object in a set
    # here Embarked.mode.values is a numpy.ndarray type (what pandas use to store strings) 
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    # The following line produces [(0, 'C'), (1, 'Q'), (2, 'S')]
    #Ports = list(enumerate(np.unique(df['Embarked'])))
    # Create dic {port(char): port(int)}
    #Ports_dict = { name : i for i, name in Ports }
    Ports_dict = {'Q': 1, 'C': 0, 'S': 2}
    # Converting port string as port int
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

    ## Age
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

if __name__=='__main__':

    ###### Data cleanup

    # TRAIN DATA
    # train_df is a data frame
    train_df = pd.read_csv(traincsvname, header=0)
    train_df = clean_data(train_df)
    #removes strings features
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # TEST DATA
    test_df = pd.read_csv(testcsvname, header=0)
    test_df = clean_data(test_df)
    ids = test_df['PassengerId'].values
    test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    ###### Convert data frame to numpy array
    train_data = train_df.values
    test_data = test_df.values

    print 'Training...'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

    print 'Predicting...'
    output = forest.predict(test_data).astype(int)

    # Writing output
    predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)    
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'
