import numpy as np


# Defining functions for learning algorithms
def weight_adaptation(train_data, weights, threshold, error_threshold, epoch=0, learning_rate=1, widrow_hoff=False):
    while True:
        epoch += 1
        #predicted = []
        true_positives = 0

        for row in train_data:
            y = row[3]
            weighted_sum = sum(weights * row[0:3])

            if weighted_sum >= threshold:
                predicted_y = 1

            else:
                predicted_y = 0
            
            #predicted.append(predicted_y)
            if widrow_hoff:
                delta = y - predicted_y
                if predicted_y == y:
                    true_positives += 1
                    continue
                elif y == 1 and predicted_y == 0:
                    weights = weights + learning_rate * delta * row[0:3]
                else:
                    weights = weights - learning_rate * row[0:3]

            else:
                if predicted_y == y:
                    true_positives += 1
                    continue
                elif y == 1 and predicted_y == 0:
                    weights = weights + learning_rate * row[0:3]
                else:
                    weights = weights - learning_rate * row[0:3]
            
        error = 1 - (true_positives/len(train_data))
        print("Epoch: ", epoch, "   True positives: ", true_positives, "(", len(train_data), ")",  "    Error: ", error)

        if error <= error_threshold:
            print("New weights: ", weights)
            break
    return weights



# Function to test learning method
def test_neural_network(new_weights, test_data):
    true_positives = 0
    for row in test_data:
        y = row[3]
        weighted_sum = sum(new_weights * row[0:3])
        if weighted_sum >= threshold:
            predicted_y = 1
        else:
            predicted_y = 0
        if y == predicted_y:
            true_positives += 1
            
    accuracy = (1 - true_positives/len(test_data)) * 100
    print("The accuracy of weight adaptaion upon test is", accuracy, "%\n")



# Initialising weights

weights = np.array([-1.0, 1.0, 1.0])  # random values
threshold = 1.0 # negative of first weight

# Initialising training data
train_data  = np.array([[1.0, 0.2, 0.8, 0],  # Data point 1 (features and label)
    [1.0, 0.7, 0.1, 1],  # Data point 2
    [1.0, 0.4, 0.5, 0],  # Data point 3
    [1.0, 0.1, 0.9, 1],  # Data point 4
    [1.0, 0.8, 0.3, 0],  # Data point 5
    [1.0, 0.5, 0.4, 1],  # Data point 6 (added)
    [1.0, 0.9, 0.7, 0],  # Data point 7 (added)
    [1.0, 0.3, 0.2, 1],  # Data point 8 (added)
    [1.0, 0.6, 0.9, 0],  # Data point 9 (added)
    [1.0, 0.2, 0.1, 1]])

# Initialising testing data
test_data = np.array([
    [1.0, 0.3, 0.6, 0],  # Data point for testing
    [1.0, 0.9, 0.2, 1],  # Data point for testing
    [1.0, 0.5, 0.7, 1],  # Data point for testing
    [1.0, 0.2, 0.4, 0]   # Data point for testing
])


# Training and testing normal method
print("Weight adaptation - Normal version")
new_weights = weight_adaptation(train_data, weights, threshold, error_threshold=0.2)
test_neural_network(new_weights, test_data)


# Training and testing modified method
print("Weight adaptation - Modified version")
new_weights = weight_adaptation(train_data, weights, threshold, error_threshold=0.2, learning_rate=0.6)
test_neural_network(new_weights, test_data)


# Training and testing Widrow Hoff method
print("Weight adaptation - Widrow Hoff rule")
new_weights = weight_adaptation(train_data, weights, threshold, error_threshold=0.2, learning_rate=0.6, widrow_hoff=True)
test_neural_network(new_weights, test_data)

'''"EVALUATION: Modified weight adaptation and Widrow Hoff delta rule yield the same results, with both them
maximising their accuracy at 6 steps. They are both more efficient that normal weight adaptation due to the 
less number of epochs required for a maximum score. They can't achieve 100% accuracy on training, but can on testing.
This is likely due to the specific data used for training and testing, as well as the initial values of the weights.'''




