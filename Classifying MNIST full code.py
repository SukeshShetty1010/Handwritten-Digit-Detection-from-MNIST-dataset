import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def image_processor(image):
    threshold = 170
    
    # Threshold the data - set pixel values below the threshold to 0 and above or equal to the threshold to 1
    binary_image = np.where(image < threshold, 0, 1)

    # Get the indices of all non-zero pixels in the image
    nonzero_indices = np.nonzero(binary_image)

    # Find the minimum and maximum x and y values of the non-zero pixels
    min_x, max_x = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_y, max_y = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])

    # Crop the image to the bounding box containing all non-zero pixels
    boxed_image = binary_image[min_y:max_y, min_x:max_x]

    # Convert the numpy array to a PIL Image object
    boxed_image = Image.fromarray(boxed_image)

    # Resize the image to 20x20
    stretched_bounding_box = boxed_image.resize((20, 20))

    # Convert the PIL Image object back to a numpy array
    stretched_bounding_box = np.array(stretched_bounding_box)

    # Return both the processed and thresholded image as numpy arrays
    return stretched_bounding_box, binary_image

#---------------------------------------- KNN on threshold images -----------------------------------

def knn_classifier_thresholded(train_images, train_labels, test_images, test_labels, k=3):
    # Reshape the training and testing images
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the training data to the k-NN model
    knn.fit(train_images, train_labels)

    # Predict the labels for the training and testing data
    train_predicted_labels = knn.predict(train_images)
    test_predicted_labels = knn.predict(test_images)

    # Calculate the accuracy of the k-NN model for the training and testing data
    train_accuracy = accuracy_score(train_labels, train_predicted_labels) * 100
    test_accuracy = accuracy_score(test_labels, test_predicted_labels) * 100
    
    print('KNN classifier accuracy for threshold images on training data: {:.2f}%'.format(train_accuracy))
    print('KNN classifier accuracy for threshold images on testing data: {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the testing data
    return test_predicted_labels

#------------------------------------------------------------------------------------------------------

#---------------------------------------- KNN on Processed Images -----------------------------------

def knn_classifier_processed(train_images, train_labels, test_images, test_labels, k=3):
    # Reshape the training and testing images
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the training data to the k-NN model
    knn.fit(train_images, train_labels)

    # Predict the labels for the training and testing data
    train_predicted_labels = knn.predict(train_images)
    test_predicted_labels = knn.predict(test_images)

    # Calculate the accuracy of the k-NN model for the training and testing data
    train_accuracy = accuracy_score(train_labels, train_predicted_labels) * 100
    test_accuracy = accuracy_score(test_labels, test_predicted_labels) * 100
    
    print('KNN classifier accuracy for processed images on training data: {:.2f}%'.format(train_accuracy))
    print('KNN classifier accuracy for processed images on testing data: {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the testing data
    return test_predicted_labels

#------------------------------------------------------------------------------------------------------

#---------------------------------------- Gaussian Naive Bayes on Orignal images -----------------------------------

def gaussian_naive_bayes_orignal_image(train_images, train_labels, test_images, test_labels):
    # Reshape the training and testing images
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Initialize the Gaussian Naive Bayes classifier
    nb = GaussianNB()

    # Fit the training data to the Gaussian Naive Bayes model
    nb.fit(train_images, train_labels)

    # Predict the labels for the training and testing data
    train_predicted_labels = nb.predict(train_images)
    test_predicted_labels = nb.predict(test_images)

    # Calculate the accuracy of the Gaussian Naive Bayes model for the training and testing data
    train_accuracy = accuracy_score(train_labels, train_predicted_labels) * 100
    test_accuracy = accuracy_score(test_labels, test_predicted_labels) * 100
    print('Gaussian Naive Bayes classifier accuracy for thresholded images (train data): {:.2f}%'.format(train_accuracy))
    print('Gaussian Naive Bayes classifier accuracy for thresholded images (test data): {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the testing data
    return test_predicted_labels

#------------------------------------------------------------------------------------------------------

#---------------------------------------- Gaussian Naive Bayes on processed images -----------------------------------

def gaussian_nb_classifier_processed(train_images, train_labels, test_images, test_labels):
    # Reshape the training and testing images
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Initialize the Gaussian Naive Bayes classifier
    nb = GaussianNB()

    # Fit the training data to the Gaussian Naive Bayes model
    nb.fit(train_images, train_labels)

    # Predict the labels for the training and testing data
    train_predicted_labels = nb.predict(train_images)
    test_predicted_labels = nb.predict(test_images)

    # Calculate the accuracy of the Gaussian Naive Bayes model for the training and testing data
    train_accuracy = accuracy_score(train_labels, train_predicted_labels) * 100
    test_accuracy = accuracy_score(test_labels, test_predicted_labels) * 100
    print('Gaussian Naive Bayes classifier accuracy for processed images (train data): {:.2f}%'.format(train_accuracy))
    print('Gaussian Naive Bayes classifier accuracy for processed images (test data): {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the testing data
    return test_predicted_labels

#------------------------------------------------------------------------------------------------------

#---------------------------------------- Bernoulli Naive Bayes on threshold images -----------------------------------

# This function applies Bernoulli Naive Bayes on thresholded images and outputs the accuracy of the classifier
def bernoulli_naive_bayes_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test):
    # Convert the thresholded images to binary values (0 or 1)
    binary_images_train = np.where(all_thresholded_images_train < 1, 0, 1)
    binary_images_test = np.where(all_thresholded_images_test < 1, 0, 1)

    # Calculate the prior probabilities for each class
    classes, counts = np.unique(y_train, return_counts=True)
    priors = counts / len(y_train)

    # Calculate the likelihoods for each pixel and each class
    likelihoods = np.zeros((len(classes), binary_images_train.shape[1], binary_images_train.shape[2]))
    for i, c in enumerate(classes):
        # Get all images in the current class and calculate the likelihoods for each pixel
        class_images = binary_images_train[y_train == c]
        likelihoods[i] = (class_images.sum(axis=0) + 1) / (len(class_images) + 2)

    # Make predictions for the train and test data
    train_predictions = []
    for image in binary_images_train:
        posterior_probs = np.zeros(len(classes))
        for i, c in enumerate(classes):
            # Calculate the posterior probability for the current class
            posterior_probs[i] = np.log(priors[i]) + np.log((likelihoods[i] ** image) * ((1 - likelihoods[i]) ** (1 - image))).sum()
        # Choose the class with the highest posterior probability as the predicted class
        train_predictions.append(classes[np.argmax(posterior_probs)])
    
    test_predictions = []
    for image in binary_images_test:
        posterior_probs = np.zeros(len(classes))
        for i, c in enumerate(classes):
            # Calculate the posterior probability for the current class
            posterior_probs[i] = np.log(priors[i]) + np.log((likelihoods[i] ** image) * ((1 - likelihoods[i]) ** (1 - image))).sum()
        # Choose the class with the highest posterior probability as the predicted class
        test_predictions.append(classes[np.argmax(posterior_probs)])

    # Calculate the accuracy of the classifier for the train and test datasets
    train_accuracy = ((train_predictions == y_train).mean()) * 100
    test_accuracy = ((test_predictions == y_test).mean()) * 100
    print('Bernoulli Naive Bayes classifier accuracy for thresholded images - Train: {:.2f}%'.format(train_accuracy))
    print('Bernoulli Naive Bayes classifier accuracy for thresholded images - Test: {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the test data
    return test_predictions


#------------------------------------------------------------------------------------------------------

#---------------------------------------- Bernoulli Naive Bayes on Proccesed images -----------------------------------

def bernoulli_naive_bayes_proccesed_images(processed_x_train, y_train, processed_x_test, y_test):
    # Convert the thresholded images to binary values (0 or 1)
    binary_images_train = np.where(processed_x_train < 1, 0, 1)
    binary_images_test = np.where(processed_x_test < 1, 0, 1)

    # Calculate the prior probabilities for each class
    classes, counts = np.unique(y_train, return_counts=True)
    priors = counts / len(y_train)

    # Calculate the likelihoods for each pixel and each class
    likelihoods = np.zeros((len(classes), binary_images_train.shape[1], binary_images_train.shape[2]))
    for i, c in enumerate(classes):
        # Get all images in the current class and calculate the likelihoods for each pixel
        class_images = binary_images_train[y_train == c]
        likelihoods[i] = (class_images.sum(axis=0) + 1) / (len(class_images) + 2)

    # Make predictions for the train and test data
    train_predictions = []
    for image in binary_images_train:
        posterior_probs = np.zeros(len(classes))
        for i, c in enumerate(classes):
            # Calculate the posterior probability for the current class
            posterior_probs[i] = np.log(priors[i]) + np.log((likelihoods[i] ** image) * ((1 - likelihoods[i]) ** (1 - image))).sum()
        # Choose the class with the highest posterior probability as the predicted class
        train_predictions.append(classes[np.argmax(posterior_probs)])
    
    test_predictions = []
    for image in binary_images_test:
        posterior_probs = np.zeros(len(classes))
        for i, c in enumerate(classes):
            # Calculate the posterior probability for the current class
            posterior_probs[i] = np.log(priors[i]) + np.log((likelihoods[i] ** image) * ((1 - likelihoods[i]) ** (1 - image))).sum()
        # Choose the class with the highest posterior probability as the predicted class
        test_predictions.append(classes[np.argmax(posterior_probs)])

    # Calculate the accuracy of the classifier for the train and test datasets
    train_accuracy = ((train_predictions == y_train).mean()) * 100
    test_accuracy = ((test_predictions == y_test).mean()) * 100
    print('Bernoulli Naive Bayes classifier accuracy for processed images - Train: {:.2f}%'.format(train_accuracy))
    print('Bernoulli Naive Bayes classifier accuracy for processed images - Test: {:.2f}%'.format(test_accuracy))

    # Return the predicted labels for the test data
    return test_predictions

#---------------------------------------- SVM on threshold images -----------------------------------

def svm_classifier_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test):
    # Flatten the training and testing images
    flattened_images_train = all_thresholded_images_train.reshape(len(all_thresholded_images_train), -1)
    flattened_images_test = all_thresholded_images_test.reshape(len(all_thresholded_images_test), -1)
    
    # Train the SVM classifier
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(flattened_images_train, y_train)
    
    # Predict the labels for the training and testing sets
    y_train_pred = clf.predict(flattened_images_train)
    y_test_pred = clf.predict(flattened_images_test)
    
    # Calculate and print the accuracy for the training and testing sets
    train_accuracy = (clf.score(flattened_images_train, y_train)) * 100
    test_accuracy = (clf.score(flattened_images_test, y_test)) * 100
    print('SVM accuracy for threshold images (train): {:.2f}%'.format(train_accuracy))
    print('SVM accuracy for threshold images (test): {:.2f}%'.format(test_accuracy))

#---------------------------------------- SVM on Processed images -----------------------------------

def svm_classifier_processed_images(processed_x_train, y_train, processed_x_test, y_test):
    # Flatten the training and testing images
    flattened_images_train = processed_x_train.reshape(len(processed_x_train), -1)
    flattened_images_test = processed_x_test.reshape(len(processed_x_test), -1)
    
    # Train the SVM classifier
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(flattened_images_train, y_train)
    
    # Predict the labels for the training and testing sets
    y_train_pred = clf.predict(flattened_images_train)
    y_test_pred = clf.predict(flattened_images_test)
    
    # Calculate and print the accuracy for the training and testing sets
    train_accuracy = (clf.score(flattened_images_train, y_train)) * 100
    test_accuracy = (clf.score(flattened_images_test, y_test)) * 100
    print('SVM accuracy for processed images (train): {:.2f}%'.format(train_accuracy))
    print('SVM accuracy for processed images (test): {:.2f}%'.format(test_accuracy))

#------------------------------------------------------------------------------------------------------

#---------------------------------------- Decision forest on threshold images -----------------------------------

def decision_forest_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test):
    accuracies_train = []
    accuracies_test = []
    for num_trees in [10, 30]:
        for max_depth in [4, 16]:
            # Train a random forest classifier on the thresholded images
            clf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth)
            clf.fit(all_thresholded_images_train.reshape(len(all_thresholded_images_train), -1), y_train)
            
            # Make predictions on the training set
            y_pred_train = clf.predict(all_thresholded_images_train.reshape(len(all_thresholded_images_train), -1))
            
            # Compute the accuracy of the predictions on training set
            accuracy_train = (accuracy_score(y_train, y_pred_train)) * 100
            accuracies_train.append(accuracy_train)
            
            # Make predictions on the test set
            y_pred_test = clf.predict(all_thresholded_images_test.reshape(len(all_thresholded_images_test), -1))
            
            # Compute the accuracy of the predictions on test set
            accuracy_test = (accuracy_score(y_test, y_pred_test)) * 100
            accuracies_test.append(accuracy_test)
            
            print(f"Number of trees: {num_trees}, Maximum depth: {max_depth}, Accuracy of thresholded images on train set: {accuracy_train:.4f}, Accuracy of thresholded images on test set: {accuracy_test:.4f}")
    return accuracies_train, accuracies_test

#------------------------------------------------------------------------------------------------------------------------

#---------------------------------------- Decision forest on processed images -----------------------------------

def decision_forest_processed_images(processed_x_train, y_train, processed_x_test, y_test):
    accuracies_train = []
    accuracies_test = []
    for num_trees in [10, 30]:
        for max_depth in [4, 16]:
            # Train a random forest classifier on the processed images
            clf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth)
            clf.fit(processed_x_train.reshape(len(processed_x_train), -1), y_train)
            
            # Make predictions on the training set
            y_pred_train = clf.predict(processed_x_train.reshape(len(processed_x_train), -1))
            
            # Compute the accuracy of the predictions on training set
            accuracy_train = (accuracy_score(y_train, y_pred_train)) * 100
            accuracies_train.append(accuracy_train)
            
            # Make predictions on the test set
            y_pred_test = clf.predict(processed_x_test.reshape(len(processed_x_test), -1))
            
            # Compute the accuracy of the predictions on test set
            accuracy_test = (accuracy_score(y_test, y_pred_test)) * 100
            accuracies_test.append(accuracy_test)
            
            print(f"Number of trees: {num_trees}, Maximum depth: {max_depth}, Accuracy of processed images on train set: {accuracy_train:.4f}, Accuracy of processed images on test set: {accuracy_test:.4f}")
    return accuracies_train, accuracies_test

#------------------------------------------------------------------------------------------------------

def process_images(images):
    processed_images = []
    all_thresholded_images = []
    for image in images:
        processed, thresholded = image_processor(image)
        processed_images.append(processed)
        all_thresholded_images.append(thresholded)
    return np.array(processed_images), np.array(all_thresholded_images)


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Process the training and testing images
processed_x_train, all_thresholded_images_train = process_images(x_train)
processed_x_test, all_thresholded_images_test = process_images(x_test)

# Apply k-NN on the thresholded images with k=3 and print the accuracy
print("Accuracy score for thresholded images")

predicted_labels = knn_classifier_thresholded(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test, k=3)

# Apply k-NN on the processed images with k=3 and print the accuracy
print("Accuracy score for processed images")

predicted_labels = knn_classifier_processed(processed_x_train, y_train, processed_x_test, y_test, k=3)

# Apply Gaussian Naive Bayes on the processed images with k=3 and print the accuracy
print("Accuracy score for thresholded images")

test_predicted_labels = gaussian_naive_bayes_orignal_image(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test)

# Apply the Gaussian Naive Bayes classifier on processed images
print("Accuracy score for processed images")

test_predicted_labels = gaussian_nb_classifier_processed(processed_x_train, y_train, processed_x_test, y_test)

# Apply Bernoulli Naive Bayes on the threshold images print the accuracy
print("Accuracy score for thresholded images")

bernoulli_naive_bayes_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test)

# Apply Bernoulli Naive Bayes on the Processed images print the accuracy
print("Accuracy score for processed images")

bernoulli_naive_bayes_proccesed_images(processed_x_train, y_train, processed_x_test, y_test)

# Apply SVM on the threshold images and print the accuracy for both the training and testing sets
svm_classifier_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test)

# Apply SVM on the processed images and print the accuracy for both the training and testing sets
svm_classifier_processed_images(processed_x_train, y_train, processed_x_test, y_test)

print("Accuracy score for thresholded images")

# Train and evaluate the random forest classifier on the thresholded images
train_acc_tf, test_acc_tf = decision_forest_thresholded_images(all_thresholded_images_train, y_train, all_thresholded_images_test, y_test)

print("Accuracy score for processed images")

# Train and evaluate the random forest classifier on the processed images
train_acc_pf, test_acc_pf = decision_forest_processed_images(processed_x_train, y_train, processed_x_test, y_test)

# Display a particular processed image and its corresponding thresholded image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(processed_x_train[3], cmap='gray')
axs[0].set_title('Processed Image')
axs[1].imshow(all_thresholded_images_train[3], cmap='gray')
axs[1].set_title('Thresholded Image')
plt.show()