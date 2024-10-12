# %% [markdown]
# # INTRODUCTION TO MACHINE LEARNING PROJECT

# %% [markdown]
# ## Imports

# %%
import numpy as np
import math 
import random
import matplotlib.pyplot as plt

# NOTES
# TODO show that the train test creation is really random, i.e that every class has 1/3 chance of being the majority class over all the dataset
# TODO make doxygen-like comments ?
# TODO confirm the experimental results with theoretical results
# TODO not sure normalization is needed for the features ?
# TODO weighting the closest points ?
# TODO improve calculations by using linear algebra 

# %% [markdown]
# ## Loading Data

# %%
def load_data(dataset_file):
    return np.loadtxt(dataset_file, delimiter=',')

data = load_data('./waveform.data')

# Splitting the features and the labels into two different arrays
features = np.array(data[:,:data.shape[1]-1])
print(features)

# Normalization of the features
def normalization(feat_to_norm) :
    new = np.empty(shape=feat_to_norm.shape)
    for i in range(feat_to_norm.shape[1]) :
        mean = np.mean(feat_to_norm[:,i])
        std = np.std(feat_to_norm[:,i])
        new[:,i] = (feat_to_norm[:,i] - mean) / std
    return new

features = normalization(feat_to_norm=features)
print(features)
    

labels = np.array(data[:,data.shape[1]-1:])

# %% [markdown]
# ###  Creating the dataset in python

# %%
def train_test_split(features, labels, train_ratio):
    assert features.shape[0] == labels.shape[0], "Error : dimensions of features and labels should be the same"

    # Randomly shuffling the dataset's indices to pick randomly the training and test examples
    shuffled_indices = np.arange(features.shape[0])
    np.random.shuffle(shuffled_indices)

    train_size = int(len(shuffled_indices) * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    x_set = features[train_indices]
    x_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    assert x_set.shape[0] == y_train.shape[0], "Error : creation of training set went wrong."
    assert x_test.shape[0] == y_test.shape[0], "Error : creation of testing set went wrong."

    return x_set, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(features, labels, 0.8)

print("Training set : "+str(x_train.shape[0])+" examples.\nTest set : "+str(x_test.shape[0])+" examples.")

print(x_train)
print(y_train)


# %% [markdown]
# ### Various functions

# %%
# Returns the euclidian distance between the vectors x1 and x2
def euclidian_distance(x1, x2):
    assert len(x1) == len(x2), "Dimensions of x1 and x2 must be the same"

    return np.sqrt(np.sum((x2 - x1) ** 2))

# Returns an array with the distances from x to every x' in x_array
def distance_array(x, x_array):
    return np.linalg.norm(x_array - x, axis=1).reshape(-1, 1)

def accuracy(nb_true, nb_total):
    return (nb_true * 100) / nb_total

# %% [markdown]
# ## KNN Algorithm

# %%
# Returns the majority class from the features amongst the k nearest neighbors of new_input 
def knn(new_input, features, labels, k):
    # Computing the distance from new_input to every x in features
    dist_array = distance_array(new_input, features)

    # Associating the distance array with their corresponding labels
    dist_array_labeled = np.hstack([dist_array, labels])

    # Sorting the array by increasing distance order (keeping the labels associated)
    sorted_indices = np.argsort(dist_array_labeled[:, 0])
    sorted_array = dist_array_labeled[sorted_indices]

    # Count the number of occurences of each class Yj among the k nearest neighbors
    label, counts = np.unique(sorted_array[:k, 1], return_counts=True, axis=0)
    results = dict(zip(label, counts))

    # Returning the majority class among the k nearest neighbors
    majority_class = max(results, key=results.get)
    return majority_class

# Returns the accuracy of the prediction over the entire x_set for k nearest neighbors
def prediction(x_set, y_set, k):
    right_predictions = 0
    total_predictions = len(x_set)

    for i in range(total_predictions) :
        # Take the example predicted out as we don't want his class taken into account
        new_x_train = np.delete(x_set, i, axis=0)
        new_y_train = np.delete(y_set, i, axis=0)

        y_pred = knn(x_set[i], new_x_train, new_y_train, k)
        y_actual = y_set[i, 0]

        if (y_pred == y_actual):
            right_predictions +=1

    accuracy = (right_predictions * 100) / total_predictions
    print("k = "+str(k)+" acc = "+str(accuracy))
    return accuracy

# %% [markdown]
# ## Tuning k by cross-validation

# %%
#Splitting the training set into x_subsets for cross-validation
def k_folds(x_training_set, y_training_set, nb_folds):
    x = np.split(x_training_set, nb_folds)
    y = np.split(y_training_set, nb_folds)
    assert len(x) == len(y), "Error creating the subsets for cross-validation"
    return x, y

nb_folds = 5
x_subsets, y_subsets = k_folds(x_train, y_train, nb_folds)

k_values = range(1, 100)
accuracies = []

for k in k_values : 
    fold_accuracies = []

    for n in range(nb_folds):
        x_val_fold = x_subsets[n]
        y_val_fold = y_subsets[n]

        fold_accuracy = (prediction(x_val_fold, y_val_fold, k))
        fold_accuracies.append(fold_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    accuracies.append(mean_accuracy)  
    print("k = "+str(k)+" mean accuracy = "+str(mean_accuracy))   

# %% [markdown]
# ### Plotting

# %%
max_accuracy = max(accuracies)
best_k = k_values[accuracies.index(max_accuracy)] 

plt.figure(figsize=(10, 6), dpi=500) 
plt.plot(k_values, accuracies, label='Accuracy', marker='o', markersize=2,linestyle='None')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
plt.axhline(y=86, color='g', linestyle='--', label=f'Bayesian accuracy')
plt.text(best_k-1, max_accuracy+.5, f'k={best_k}, Acc={max_accuracy:.2f}', 
         horizontalalignment='right', verticalalignment='bottom', 
         color='red', fontsize=12)
plt.xlabel('k values')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, max(k_values)+2, 50)) 
plt.grid(True)
plt.legend()

plt.show()

print(accuracies[46])

# %%
#test prediction sur test set
print(prediction(x_test, y_test, 82))


