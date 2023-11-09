import numpy as np
from collections import Counter

class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self, model):
        """Initialises a new classifier."""
        self.model = model
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        # YOUR CODE HERE
        class_likelihoods = {}

        for class_name, class_data in self.model.items():
            if class_name != 'vocabulary':
                class_likelihood = np.log(class_data['class_likelihood'])
                for word in x:
                    if word in class_data['word_probs']:
                        class_likelihood += np.log(class_data['word_probs'][word])
            
                class_likelihoods[class_name] = class_likelihood
            
        
        predicted_class = max(class_likelihoods, key=class_likelihoods.get)
        
        return predicted_class
        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################
        # YOUR CODE HERE
        
        class_counts = {}
        word_counts = {}
        possible_words = set()
        
        for instance, label in data:
            if label not in class_counts:
                class_counts[label] = 0
                word_counts[label] = {}
            
            class_counts[label] += 1
            
            for word in instance:
                possible_words.add(word)
                if word not in word_counts[label]:
                    word_counts[label][word] = 1
                else:
                    word_counts[label][word] += 1
                    
        
        model = {}

        for class_name, class_count in class_counts.items():
            class_likelihood = class_count / len(data)
            word_probs = {}
            
            not_seen = possible_words - set(word_counts[class_name].keys())
            
            if k != 0:
                for not_seen_word in not_seen:
                    word_counts[class_name][not_seen_word] = 0

            total_word_count = sum(word_counts[class_name].values())
            vocab = len(possible_words)

            for word, count in word_counts[class_name].items():
                word_probs[word] = (count + k)  / (total_word_count + k * vocab)

            model[class_name] = {'class_likelihood': class_likelihood, 'word_probs': word_probs}
        
        return cls(model)
    
        ############################################################



def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    
    total_counts = Counter()
    
    for tweet, label in data:
        total_counts.update(tweet)
                
    # Sort the items in descending order based on counts
    total_counts = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True))
    
    # Keep the top (most frequent) 50 words
    for key in list(total_counts)[30:]:
        del total_counts[key]
                
    features = set(total_counts.keys())
    
    return features
    ##################################################################


def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    ##################################################################

