"""Classification system.
Classifies letters from an image using
nearest neighbour classifier and reducing 
to 10 dimentions with pca

author:Iulian Adrian Lazarina

version: v1.0
"""


import numpy as np
import utils.utils as utils
import scipy.linalg
from PIL import Image
from random import randint
from scipy.ndimage.filters import median_filter


def pca_feature_reduction(feature_vector):
    """pca reduces to 10 features
       picking different features gives different results
    parameters:
     feature_vector: array of feature vectors
    
    Returns: eigenvector matrix
    """

    covx = np.cov(feature_vector, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 11, N - 2))#can modify the picked features by changing these values
    v = np.fliplr(v)

    return v



def divergence(class1, class2):
    """compute a vector of 1-D divergences

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12


def reduce_dimensions(feature_vectors_full, model):
    """apply pca to feature vectors
    
    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    pca_vector = model['pca_vector']
    pcadata = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), pca_vector)

    return pcadata


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    c=0
    for page_name in train_page_names:
        if c<2:#can adjust to how many pages we add noise
            images_train = add_noise(utils.load_char_images(page_name, images_train))#adds noise to training pages
            labels_train = utils.load_labels(page_name, labels_train)
            c=c+1
        else:
            images_train = utils.load_char_images(page_name, images_train)
            labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    model_data['pca_vector'] = pca_feature_reduction(fvectors_train_full).tolist() #add pca_vector to model
    
    print('Reducing to 10 dimensions')
   
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
    

    model_data['fvectors_train'] = fvectors_train.tolist()

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """

    bbox_size = model['bbox_size']
    images_test = reduce_noise(utils.load_char_images(page_name))#reduce noise on all test pages


    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced



def reduce_noise(image):
    """reduces the noise of an image using median_filter
    Param:
      image: the image to which we apply the filter

    Returns: filtered page
    """
    for n in range(len(image)):
        image[n] = median_filter(image[n],3)
    return image


def add_noise(page):
    """adds noise to a page is a pseudo-random way
        the noise will always be added in the same way
        because the seed is set here and it stays the same
    Param:
      page: the page with the noise added
    """
    page = np.array(page)
    length = page.shape[0]
    np.random.seed(3435)#set the seed so that we add the same noise every time in order to get a consistent result
    noise = np.random.randn(length)
    noise = noise.reshape(length)        
    new_page = page + page * noise
    return new_page.tolist()



def classify(train, train_labels, test, features=None):
    """Perform nearest neighbour classification."""

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour 
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance

    nearest = np.argmax(dist, axis=1)
    label = train_labels[nearest]

    return label


def classify_page(page, model):

    """classifies a page by calling classify

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

   
    return classify(fvectors_train, labels_train, page)


def correct_errors(page, labels, bboxes, model):
    """
    By storing a dictionary into our model I should be able to compare
    words that I find when classifying with words from the dictionary
    By searching and comparing I can flag one letter differences 
    and correct them by replacing the wrong letter with second closest


    This approach however does have it's disadvantages
    Mainly the fact that since lost of words are quite similar the 
    wrong letter might be considered correct and a letter that was classified corectly might 
    be flaged as wrong and be replaced resulting in a loss of score and accuracy


    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels
