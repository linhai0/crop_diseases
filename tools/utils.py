import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
def historm(x):
    sns.distplot(x)#, kde=False, rug=True)
    plt.show()

def sns_confusion_matrix(x,y):
    mat = confusion_matrix(x, y)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=x.target_names,
            yticklabels=x.target_names)
    plt.ylabel('predicted label')
