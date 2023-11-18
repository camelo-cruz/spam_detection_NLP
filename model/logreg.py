import warnings
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
from nltk.stem import WordNetLemmatizer


warnings.filterwarnings("ignore", category=RuntimeWarning)


class LogReg:
    def __init__(self, eta=0.01, num_iter=30, lambda_value=0.1):
        self.eta = eta
        self.num_iter = num_iter
        self.lambda_value = lambda_value

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution
        return np.exp(inputs) / np.sum(np.exp(inputs))


    def train(self, X, Y):
        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.random.randn(X.shape[1], Y.shape[1])
        self.biases = np.zeros(Y.shape[1])
        
        print(f'training logistic regression for {self.num_iter} epochs with learning rate {self.eta} '
              f'and regularization lambda {self.lambda_value}\n')
        for i in range(self.num_iter):
            
            Y_hat = self.p(X)
            
            weight_gradient = np.dot(X.T, (Y_hat - Y)) / len(X)
            bias_gradient = np.mean(Y_hat - Y, axis=0).values
            
            l2_reg_term = (self.lambda_value * np.square(self.weights)) / len(X)
            
            self.weights -= self.eta * (weight_gradient + l2_reg_term)
            self.biases -= self.eta * bias_gradient
            
            
            new_probs = self.p(X)
            loss = - np.sum(Y * np.log(new_probs), axis=1).mean()
            correct = np.sum(np.round(new_probs) == Y).mean()
            accuracy = correct / len(Y) 
            
            print(f'epoch {i+1}\n accuracy = {accuracy} epoch_loss = {loss}')
    
        return None
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


def featurize(data, train_data=None, preprocessing=False):
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
    
    #some optional preprocessing for eliminating features == words
    if preprocessing:
        lemmatizer = WordNetLemmatizer()
        postprocessed_data = sorted([lemmatizer.lemmatize(word) for sentence, label in train_data for word in sentence])
        postprocessed_data = postprocessed_data[30:]
    else:
        postprocessed_data = [word for sentence, label in train_data for word in sentence]
        
    vocab = {word:"" for word in postprocessed_data}
    mapping = buildw2i(vocab)
    
    #Put the words in order
    vocab_in_int = sorted(np.array(list(mapping.values())))
    
    X_data = []
    Y_data = []

    #Create the matrix with one hot encoding
    for sentence, label in data:
        #initialiting a matrix of zeros with length vocabulary
        one_hot_sentence = np.zeros(len(vocab_in_int))
        for word in sentence:
            if preprocessing:
                word = lemmatizer.lemmatize(word)
            if word in mapping:
                word_index = mapping[word]
                one_hot_sentence[word_index] += 1

        X_data.append(one_hot_sentence)
        if label == 'offensive':
            Y_data.append([1, 0])  
        else:
            Y_data.append([0, 1])

    X = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(X_data))
    Y = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(Y_data))
    

    return X, Y
    ##############################################################



