def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    
    # true_labels = [0,1,1,1,1,0,0,1,0]
    # predicted_labels = [0,1,1,1,1,0,0,0,0]
    
    
    X = [tupla[0] for tupla in data]
    
    true_labels = [tupla[1] for tupla in data]
    predicted_labels = [classifier.predict(x) for x in X]
    
    
    correct = sum([1 for p, t in zip(predicted_labels, true_labels) if p == t])
    total = len(predicted_labels)    
    
    return float(correct/total)
    ################################################################


def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    
    X = [tupla[0] for tupla in data]
    
    true_labels = [tupla[1] for tupla in data]
    predicted_labels = [classifier.predict(x) for x in X]
    
    tp = sum((p == 'offensive' and t == 'offensive') for p, t in zip(predicted_labels, true_labels))
    fp = sum((p == 'offensive' and t == 'nonoffensive') for p, t in zip(predicted_labels, true_labels))
    fn = sum((p == 'nonoffensive' and t == 'offensive') for p, t in zip(predicted_labels, true_labels))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1_score = float(2 * precision * recall) / (precision + recall)
    
    
    return f1_score
    ################################################################


