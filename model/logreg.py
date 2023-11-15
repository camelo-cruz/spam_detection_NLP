import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        
        # result = []

        # for element in inputs:
        #     new_element = []
        #     exp_values = np.exp(element)
        #     sum_exp_values = sum(exp_values)

        #     for value in exp_values:
        #         new_element.append(value / float(sum_exp_values))

        #     result.append(new_element)
        
        return np.exp(inputs) / np.sum(np.exp(inputs))


    def train(self, X, Y):
        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.random.randn(X.shape[1], Y.shape[1])
        self.biases = np.zeros(Y.shape[1])
        
        
        print(f'training logistic regression for {self.num_iter} epochs\n')
        for i in range(self.num_iter):
            random_idx = np.random.choice(X.shape[0], 100, replace=False)
            mini_batch = X.iloc[random_idx]
            real_labels = Y.iloc[random_idx]
    
            predicted_labels = self.p(mini_batch)
            
            l2_regularization = 0.5 * 0.01 * np.sum(self.weights**2)
            
            loss = -np.sum(real_labels * np.log(predicted_labels), axis=1).mean() + l2_regularization
            
            gradient = np.dot(mini_batch.T, (predicted_labels - real_labels)) + (0.01 * self.weights)
            self.weights -= self.eta * gradient
            self.biases -= self.eta * np.sum((predicted_labels - real_labels), axis=0)
            self.biases = self.biases.values
            
            print(f'epoch {i+1}\n epoch_loss = {loss}')
    
        return self.weights, self.biases
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        linear_results = np.dot(X, self.weights) + self.biases

        probs = []
        for values in linear_results:       
            probs.append(self.softmax(values))
        
        return np.array(probs)
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        
        probs = self.p(X)
        
        predictions = np.argmax(probs, axis=1)
        predictions_one_hot = pd.get_dummies(predictions)
    
        return pd.DataFrame(predictions_one_hot)
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



