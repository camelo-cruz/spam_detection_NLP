from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import sparse



class LogReg:
    def __init__(self, eta=0.01, num_iter=30):
        self.eta = eta
        self.num_iter = num_iter

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution
        return np.exp(inputs) / float(sum(np.exp(inputs)))


    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            # YOUR CODE HERE
            #     TODO:
            #         1) Fill in iterative updating of weights
            pass
        return None
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        results = np.dot(X, self.weights)
        
        return results
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        linear_results = self.p(X)
        probs = self.softmax(linear_results)
        
        predictions = []
        for value in probs:
            if value > 0.5: #threshhold for positive class
                predictions.append([1, 0])
            else:
                predictions.append([0, 1])
                
        
        return predictions
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    
    mapping = defaultdict(int)
    index = 0
    for word in vocab:
        mapping[word] = index
        index += 1
    return mapping
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION ####################### 
    vocab = {word:"" for sentence, label in train_data for word in sentence}
    mapping = buildw2i(vocab)

    vocab_in_int = sorted(np.array(list(mapping.values())))
    
    X_data = []
    Y_data = []

    for sentence, label in data:
        one_hot_sentence = np.zeros(len(vocab_in_int))
        for word in sentence:
            if word in mapping:
                word_index = mapping[word]
                one_hot_sentence[word_index] = 1

        X_data.append(one_hot_sentence)
        if label == 'offensive':
            Y_data.append([1, 0])  
        else:
            Y_data.append([0, 1])

    X = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(X_data))
    Y = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(Y_data))
    

    return X, Y
    ##############################################################



