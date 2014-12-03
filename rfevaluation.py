#================================================================================
#
# Some code evaluatin random forest
# 
# 
#
#================================================================================
import rforest
import matplotlib.pyplot as plt

def scoreVSnsamples():
    """
    Plots the score vs the number of sample
    """
    csvfname = '/Users/jean-francoisrajotte/projects/kaggle/titanic/data/train.csv'
    accuracy_mean_list = []
    accuracy_std_list = []
    #fraction_list = [0.5, 1.0]
    fraction_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ifraction in fraction_list:
        irfmodel = rforest.RandomForestModel(csvfname)
        itrain_dic = irfmodel.trainNselfCheck(ifraction)
        iscores = itrain_dic['scores']
        accuracy_mean_list.append(iscores.mean())
        accuracy_std_list.append(iscores.std())
    print accuracy_mean_list

    fig, axs = plt.subplots(nrows=1, ncols=1)
    #ax = axs[0, 1]
    axs.errorbar(fraction_list, accuracy_mean_list, yerr=accuracy_std_list, fmt='o')
    axs.set_title('Accuracy vs sample fraction')
    plt.xlim(0, 1.05)
    plt.show()

if __name__=='__main__':
    scoreVSnsamples()
