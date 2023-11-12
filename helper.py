import os
import matplotlib.pyplot as plt
import numpy as np

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    
    folder_path = 'plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    values = np.linspace(2, 0, num=10)
    accs = []
    f1 = []
    
    for value in values:
        nb = NaiveBayes.train(train_data, value)
        accs.append(accuracy(nb, test_data))
        f1.append(f_1(nb, test_data))
    
    # Plot both accuracy and F1 score on the same plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy with blue color and label
    plt.plot(values, accs, marker='o', color='blue', label='Accuracy')
    
    # Annotate each point with its corresponding accuracy value
    for i, txt in enumerate(accs):
        plt.text(values[i], txt, f'{txt:.2f}', ha='center', va='bottom', color='blue')
    
    # Plot F1 score with orange color and label
    plt.plot(values, f1, marker='o', color='orange', label='F1 Score')
    
    # Annotate each point with its corresponding F1 score value
    for i, txt in enumerate(f1):
        plt.text(values[i], txt, f'{txt:.2f}', ha='center', va='bottom', color='orange')
    
    plt.title('Accuracy and F1 Score vs. Smoothing Value')
    plt.xlabel('Parameter Value')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    
    # Save the plot to a file
    plt.savefig(os.path.join(folder_path, 'smoothing_vs_scores_plot.png'))
    plt.show()
    
    #####################################################################


def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    
    without_common = features1(train_data)
    print("---- Training naive bayes classifier without common words")
    nb = NaiveBayes.train(without_common)
    print("Accuracy: ", accuracy(nb, test_data))
    print("F_1: ", f_1(nb, test_data))
    
    
    lematized_train_data = features2(train_data)
    print("---- Training naive bayes classifier with lemmas ")
    lematized_nb = NaiveBayes.train(lematized_train_data)
    print("Accuracy: ", accuracy(lematized_nb, test_data))
    print("F_1: ", f_1(lematized_nb, test_data))
    
    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################
