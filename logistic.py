from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


print(__doc__)

# packup 49 subviews
X = []
x_temp = []
xx = 1

# abstract how many in one LF image
sub_num_train = 1
train_num = 70*sub_num_train
y = []
for i in range(12):
    if i == 0 or i == 2 or i ==4:
        y = y + [0] * train_num
    else:
        y = y + [i] * train_num

print (set(y))

#sys.exit("=====tyler======")

with open('train_patch32.csv') as csvfile:
#with open('train_4combiF1.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x_temp = x_temp + list(map(float, row))
            if (xx%5 == 0):
                '''
                print ("np.array(x_temp)", np.array(x_temp).shape)
                x_temp = np.split(np.array(x_temp),20480)
                print ("np.array(x_temp)", np.array(x_temp).shape)
                x_temp = pca.fit(x_temp)
                print ("np.array(x_temp)", np.array(x_temp).shape)
                sys.exit("=====tyler======")
                '''
                X.append(x_temp)
                x_temp = []
            xx += 1
            #X.append(list(map(float, row)) + list(map(float, row[0:4096])))
            #X.append(list(map(float, row[0:4096]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288]))+list(map(float, row[112288:16384]))+list(map(float, row[0:4096])))
            #X.append(list(map(float, row))+list(map(float, row[0:4096]))+list(map(float, row[8192:16384]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288]))+list(map(float, row[0:4096])))
            #X.append(list(map(float, row))+list(map(float, row[0:4096]))+list(map(float, row[8192:16384]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288])))


# ============================
X_test = []
sub_num_test = 1
test_num = 30*sub_num_test  #30 is from 70/30; 4 is formal test imgs
xx = 1

y_ground = []
for i in range(12):
    if i == 0 or i == 2 or i ==4:
        y_ground = y_ground + [0] * test_num
    else:
        y_ground = y_ground + [i] * test_num

print (set(y_ground))

# packup 49 subviews
with open('test_patch32.csv') as csvfile:
#with open('test_4combiF1.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x_temp = x_temp + list(map(float, row))
            if (xx%5 == 0):
                X_test.append(x_temp)
                x_temp = []
            xx += 1
            #X_test.append(list(map(float, row)) + list(map(float, row[0:4096])))
            #X_test.append(list(map(float, row[0:4096]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288]))+list(map(float, row[112288:16384]))+list(map(float, row[0:4096])))
            #X_test.append(list(map(float, row))+list(map(float, row[0:4096]))+list(map(float, row[8192:16384]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288]))+list(map(float, row[0:4096])))
            #X_test.append(list(map(float, row))+list(map(float, row[0:4096]))+list(map(float, row[8192:16384]))+list(map(float, row[4096:8192]))+list(map(float, row[8192:112288])))
X_train = np.array(X)
X_test = np.array(X_test)
y_train = np.array(y)
y_test = np.array(y_ground)


h = 160
w = 128
n_components = 1024

target_names = np.array(["Fabric", "Foliage", "Fur", "Glass", "Leather", "Metal", "Paper", "Plastic", "Sky", "Stone", "Water", "Wood"])
print ("target_names",target_names.shape)
n_classes = target_names.shape[0]
print ("n_classes",target_names.shape)

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.25, random_state=42)

print ("X_train",X_train.shape)
print ("X_test",X_test.shape)
print ("y_train",y_train.shape)
print ("y_test",y_test.shape)
#sys.exit("===================================")

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

'''
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
#print ("pca",np.array(pca).shape)

eigenfaces = pca.components_.reshape((n_components, h, w))
print ("eigenfaces",eigenfaces.shape)
#sys.exit("===================================")

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print ("X_train_pca",X_train_pca.shape)
print ("X_test_pca",X_test_pca.shape)
#sys.exit("===================================")

'''

# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
'''
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
'''

t0 = time()
param_grid = {'C': [1e1],
              'tol': [1e-3], }
clf = GridSearchCV(LogisticRegression(), param_grid)
#clf = LinearSVC(random_state=0)

#clf = clf.fit(X_train_pca, y_train)
clf = clf.fit(X_train,y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

#clf = LinearSVC(random_state=0)
#clf.fit(X, y)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
#y_pred = clf.predict(X_test_pca)
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

my_dict = {1:'Fabric', 2:'Foliage', 3:'Fur', 4:'Glass', 5:'Leather', 6:'Metal', 7:'Paper', 8:'Plastic', 9:'Sky', 10:'Stone', 11:'Water', 12:'Wood'}
for i in range(12):
    print(my_dict[i+1], accuracy_score(y_test[(i*30):(i+1)*30], y_pred[(i*30):(i+1)*30]))
print (accuracy_score(y_ground, y_pred))

'''

# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

'''
