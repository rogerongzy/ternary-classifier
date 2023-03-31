import scipy.io
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


def load_data():
    Data_train = scipy.io.loadmat('Data_Train.mat')
    Data_train = Data_train['Data_Train']
    Label_train = scipy.io.loadmat('Label_Train.mat')
    Label_train = Label_train['Label_Train'].ravel()
    Data_test = scipy.io.loadmat('Data_test.mat')
    Data_test = Data_test['Data_test']
    return Data_train, Label_train, Data_test


# Bayes Decision Rule classifier from Scikit-learn
def BDR(Data_train, Label_train):
    bdr = GaussianNB()
    bdr.fit(Data_train, Label_train)
    return bdr

# Bayes Decision Rule classifier
class BDR2:
    def __init__(self, Data_train, Label_train):
        # shifted as array indices 
        Label_train = Label_train - 1
        classes = np.unique(Label_train)
        n_classes = len(classes)
        
        # prior probability
        self.class_priors = np.zeros(n_classes)
        for cl in classes:
            self.class_priors[cl] = len(Label_train[Label_train == cl]) / len(Label_train)

        # likelihood function
        self.class_likelihoods = []
        for cl in classes:
            Data_train_c = Data_train[Label_train == cl]
            mean = np.mean(Data_train_c, axis=0)
            cov = np.cov(Data_train_c.T)
            mvn = multivariate_normal(mean=mean, cov=cov)
            self.class_likelihoods.append(mvn)

    def predict(self, Data_test):
        n_samples = Data_test.shape[0]
        n_classes = len(self.class_priors)
        posteriors = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            posteriors[:, i] = self.class_likelihoods[i].logpdf(Data_test) + np.log(self.class_priors[i])
        return np.argmax(posteriors, axis=1) + 1 # shift back


# Fisher discriminant analysis (FDA, a.k.a. LDA) classifier from Scikit-learn
def LDA(Data_train, Label_train):
    lda = LinearDiscriminantAnalysis(n_components=2)
    x_lda = lda.fit_transform(Data_train, Label_train)
    return lda

# Fisher discriminant analysis classifier
class LDA2:
    def __init__(self, Data_train, Label_train):
        self.data = Data_train
        self.labels = Label_train - 1 # shifted as array indices
        self.means = self._calculate_mean(self.data, self.labels)
        self.Sw = self._calculate_Sw(self.data, self.labels)
        self.Sb = self._calculate_Sb(self.data, self.labels)
        self.w = self._calculate_eigen(self.Sw, self.Sb)

    def _calculate_mean(self, data, labels):
        means = []
        for i in range(3):
            means.append(np.mean(data[labels==i], axis=0))
        return means

    # within-class scatter matrix
    def _calculate_Sw(self, data, labels):
        Sw = np.zeros((4, 4))
        for i in range(labels.shape[0]):
            x = data[i]
            u = self.means[labels[i]]
            diff = (x - u).reshape(4, 1)
            Sw += np.dot(diff, diff.T)
        return Sw

    # between-class scatter matrix
    def _calculate_Sb(self, data, labels):
        Sb = np.zeros((4, 4))
        for i in range(3):
            u = self.means[i].reshape(4, 1)
            diff = (u.ravel() - np.mean(data, axis=0)).reshape(4, 1)
            Sb += np.dot(diff, diff.T)
        return Sb

    def _calculate_eigen(self, Sw, Sb):
        # generalized eigenvalues and eigenvectors
        Swinv = np.linalg.inv(Sw)
        SwinvSb = np.dot(Swinv, Sb)
        eigvals, eigvecs = np.linalg.eig(SwinvSb)
        k = 2 # former
        idx = eigvals.argsort()[::-1][:k]
        w = eigvecs[:, idx]
        return w

    def _data_transform(self, data):
        data_transformed = np.dot(data, self.w)
        return data_transformed

    def predict(self, Data_test):
        classes = np.array([], dtype=int)
        for data in Data_test:
            data_transformed = np.dot(data, self.w)
            distances = []
            for i in range(3):
                u = np.dot(self.means[i], self.w)
                distance = np.linalg.norm(data_transformed - u)
                distances.append(distance)
            classes = np.append(classes, np.argmin(distances))
        classes = classes + 1 # shift back
        return classes


# Decision Trees classifier from Scikit-learn
def DT(Data_train, Label_train):
    dt = DecisionTreeClassifier()
    dt.fit(Data_train, Label_train)
    return dt

# Decision Trees classifier
class Node:
    def __init__(self, split_feature=None, split_value=None, left=None, right=None, label=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.label = label

class DT2:
    def __init__(self, Data_train, Label_train, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = len(np.unique(Label_train))
        self.tree = self._grow_tree(Data_train, Label_train)

    def _grow_tree(self, Data_train, Label_train, depth=0):
        n_samples, n_features = Data_train.shape
        if depth == self.max_depth or n_samples < self.min_samples_split or len(np.unique(Label_train)) == 1:
            label = np.argmax(np.bincount(Label_train))
            return Node(label=label)
        best_split_feature, best_split_value, best_gain = None, None, 0
        for i in range(n_features):
            for value in np.unique(Data_train[:, i]):
                left_indices = Data_train[:, i] < value
                right_indices = Data_train[:, i] >= value
                if len(Label_train[left_indices]) > 0 and len(Label_train[right_indices]) > 0:
                    gain = self._information_gain(Label_train, Label_train[left_indices], Label_train[right_indices])
                    if gain > best_gain:
                        best_split_feature, best_split_value, best_gain = i, value, gain
        if best_gain == 0:
            label = np.argmax(np.bincount(Label_train))
            return Node(label=label)
        left_indices = Data_train[:, best_split_feature] < best_split_value
        right_indices = Data_train[:, best_split_feature] >= best_split_value
        left_subtree = self._grow_tree(Data_train[left_indices], Label_train[left_indices], depth + 1)
        right_subtree = self._grow_tree(Data_train[right_indices], Label_train[right_indices], depth + 1)
        return Node(split_feature=best_split_feature, split_value=best_split_value, left=left_subtree, right=right_subtree)
    
    def _information_gain(self, Label_train, label_left, label_right):
        p = len(label_left) / len(Label_train)
        H_total = self._entropy(Label_train)
        H_left = self._entropy(label_left)
        H_right = self._entropy(label_right)
        return H_total - p * H_left - (1 - p) * H_right
    
    def _entropy(self, label):
        p = np.bincount(label) / len(label)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    def predict(self, Data_test):
        return np.array([self._traverse_tree(data, self.tree) for data in Data_test])
    
    def _traverse_tree(self, data, node):
        if node.label is not None:
            return node.label
        if data[node.split_feature] < node.split_value:
            return self._traverse_tree(data, node.left)
        else:
            return self._traverse_tree(data, node.right)


if __name__ == '__main__':
    Data_train, Label_train, Data_test = load_data()

    clf_LDA = LDA(Data_train, Label_train)
    clf_LDA2 = LDA2(Data_train, Label_train)
    clf_BDR = BDR(Data_train, Label_train)
    clf_BDR2 = BDR2(Data_train, Label_train)
    clf_DT = DT(Data_train, Label_train)
    clf_DT2 = DT2(Data_train, Label_train)

    print('LDA:  ', clf_LDA.predict(Data_test))
    print('LDA2: ', clf_LDA2.predict(Data_test))
    print('BDR:  ', clf_BDR.predict(Data_test))
    print('BDR2: ', clf_BDR2.predict(Data_test))
    print('DT:   ', clf_DT.predict(Data_test))
    print('DT2:  ', clf_DT2.predict(Data_test))

    # LDA:  [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    # LDA2: [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    # BDR:  [1 1 1 1 1 1 1 1 1 1 3 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    # BDR2: [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    # DT:   [1 1 1 1 1 1 1 1 1 1 3 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    # DT2:  [1 1 1 1 1 1 1 1 1 1 3 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]