import math

import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from scipy.stats import rankdata
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier


class Node:
    """
    A node in a decision tree.

    Attributes
    ----------
    data : array-like
        The subset of data points that belong to this node.

    feature : int
        The index of the feature used for splitting this node.

    val_split : float
        The value used for splitting the feature at this node.

    impurity : float
        The impurity of the node.

    probabilities : array-like
        The class probabilities associated with this node.

    """

    def __init__(self, data, feature, val_split, impurity, probabilities):
        """
        Initializes a Node object with the given data and attributes.

        Parameters
        ----------
        data : array-like
            The subset of data points that belong to this node.

        feature : int
            The index of the feature used for splitting this node.

        val_split : float
            The value used for splitting the feature at this node.

        impurity : float
            The impurity of the node.

        probabilities : array-like
            The class probabilities associated with this node.
        """
        self.data = data
        self.feature = feature
        self.val_split = val_split
        self.entropy = impurity
        self.probabilities = probabilities
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return (f"Node(data={self.data}, feature={self.feature}, val_split={self.val_split}, entropy={self.entropy}, "
                f"probabilities={self.probabilities})")


class SSLTree(ClassifierMixin):
    """A decision tree classifier.

    Constructs the tree by computing the dataset's impurity using the method proposed by Levatic et al. (2017).

    Parameters
    ----------
    w : float, default=0.75
        Controls the amount of supervision. Higher values for more supervision.

    splitter : {'best', 'random'}, default='best'
        The strategy used to choose the split at each node.
        - 'best': Choose the best split based on impurity.
        - 'random': Choose the best random split.

    max_depth : int, default=4
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {'auto', 'sqrt', 'log2', int or float}, default='auto'
        The number of features to consider when looking for the best split:
        - 'auto': All features are considered.
        - 'sqrt': The square root of the total number of features.
        - 'log2': The logarithm base 2 of the total number of features.
        - int: The number of features to consider at each split.
        - float: A fraction of the total number of features to consider at each split.

    Attributes
    ----------
    w : float
        The value of the 'w' parameter.

    splitter : {'best', 'random'}
        The strategy used to choose the split at each node.

    max_depth : int, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {'auto', 'sqrt', 'log2', int or float}
        The number of features to consider when looking for the best split.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> from your_module import SSLTree

    >>> # Load iris dataset
    >>> iris = load_iris()
    >>> X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    >>> # Train SSLTree model
    >>> clf = SSLTree(w=0.75, max_depth=5)
    >>> clf.fit(X_train, y_train)

    >>> # Predict
    >>> y_pred = clf.predict(X_test)

    >>> # Evaluate accuracy
    >>> accuracy = accuracy_score(y_test, y_pred)
    >>> print("Accuracy:", accuracy)
    """

    def __init__(self, w=0.75, splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features='auto'):
        self.w = w
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
        self.total_var = None
        self.total_gini = None
        self.labels = None
        self.feature_names = None

    def _gini(self, labels):
        probs = np.unique(labels, return_counts=True)[1] / len(labels)
        return sum([-p * np.log2(p) for p in probs if p > 0])

    def _var(self, X_i):
        return (np.sum(np.square(X_i)) - np.square(np.sum(X_i)) / len(X_i)) if len(X_i) > 1 else 0

    def _entropy_ssl(self, partitions):
        subsets_labelled = [subset[subset[:, -1] != -1] for subset in partitions]

        total_count_labelled = np.sum([len(subset) for subset in subsets_labelled])
        if total_count_labelled != 0:
            gini = np.sum(
                [self._gini(subset[:, -1]) * (len(subset) / total_count_labelled) for subset in subsets_labelled])
        else:
            gini = 0

        total_count = np.sum([len(subset) for subset in partitions])
        var = 0
        for i in range(partitions[0].shape[1] - 1):
            num = 0
            for subset in partitions:
                num += self._var(subset[:, i]) * (len(subset) / total_count)

            var += num / self.total_var[i]

        return self.w * gini / self.total_gini + ((1 - self.w) / (partitions[0].shape[1] - 1)) * var

    def _split(self, data, feature, feature_val):
        mask = data[:, feature] <= feature_val
        left = data[mask]
        right = data[~mask]

        return left, right

    def _feature_selector(self, num_features):
        if self.max_features == "auto":
            max_features = num_features
        elif self.max_features == "sqrt":
            max_features = int(math.sqrt(num_features))
        elif self.max_features == "log2":
            max_features = int(math.log2(num_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, num_features)
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * num_features)
        else:
            raise ValueError("Invalid value for max_features")

        return np.random.choice(num_features, max_features, replace=False)

    def _best_split(self, data):
        best_entropy = float('inf')
        best_feature = -1
        best_feature_val = -1

        selected_features = self._feature_selector(data.shape[1] - 1)

        for feature in selected_features:
            # possible_partitions = np.percentile(data[:, feature], q=np.arange(25, 100, 25))
            possible_partitions = np.unique(data[:, feature])
            if self.splitter != 'random':
                partition_values = possible_partitions
            else:
                # https://stackoverflow.com/questions/46756606/what-does-splitter-attribute-in-sklearns-decisiontreeclassifier-do
                partition_values = [np.random.choice(possible_partitions)]

            for feature_val in partition_values:
                left, right = self._split(data, feature, feature_val)
                entropy = self._entropy_ssl([left, right])
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_feature_val = feature_val
                    best_left, best_right = left, right

        return best_left, best_right, best_feature, best_feature_val, best_entropy

    def _node_probs(self, data):

        labels_in_data = data[:, -1]

        # Only labelled data counts
        total_labels = len(labels_in_data[labels_in_data != -1])
        probs = [0] * len(self.labels)

        for i, label in enumerate(self.labels):
            label_appearances = np.where(labels_in_data == label)[0]
            if label_appearances.shape[0] > 0:
                probs[i] = label_appearances.shape[0] / total_labels

        return probs

    def _create_tree(self, data, depth):

        if self.max_depth is not None and depth > self.max_depth:
            return None

        left_data, right_data, feature, feature_val, entropy = self._best_split(data)

        left_data_labelled = left_data[left_data[:, -1] != -1]
        right_data_labelled = right_data[right_data[:, -1] != -1]

        if len(left_data_labelled) == 0 and len(right_data_labelled) == 0:
            return None

        root = Node(data, feature, feature_val, entropy, self._node_probs(data))

        if self.min_samples_leaf >= len(left_data_labelled) and self.min_samples_leaf >= len(right_data_labelled):
            return root

        if 1.0 in root.probabilities:
            return root

        # Minimum number of samples required to split an internal node.
        if (len(left_data_labelled) + len(right_data_labelled)) >= self.min_samples_split:
            root.left = self._create_tree(left_data, depth + 1)
            root.right = self._create_tree(right_data, depth + 1)
        else:
            root.left = None
            root.right = None

        return root

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        all_labels = np.unique(y)

        # Unlabelled samples must have -1 label
        self.labels = np.sort(all_labels[all_labels != -1])

        data = np.concatenate((X, y[:, np.newaxis]), axis=1)

        self.total_gini = self._gini(data[data[:, -1] != -1][:, -1])
        self.total_var = [self._var(data[:, i]) for i in range(data.shape[1] - 1)]

        self.tree = self._create_tree(data, 0)

        return self

    def single_predict(self, x):
        # Starts on root
        node = self.tree

        predictions = [0] * self.labels
        while node:  # Until leaf is reached
            predictions = node.probabilities
            if x[node.feature] <= node.val_split:
                node = node.left
            else:
                node = node.right

        return predictions

    def predict_proba(self, X):
        return [self.single_predict(x) for x in X]

    def predict(self, X):
        return self.labels[np.argmax(self.predict_proba(X), axis=1)]

    def text_tree(self, node, depth):
        cadena = ""
        cadena += ("|" + " " * 3) * depth
        cadena += "|--- "

        if not node.left or not node.right:
            clases, cantidad = np.unique(node.data[:, -1], return_counts=True)
            return cadena + "class: " + str(self.labels[np.argmax(node.probabilities)]) + " Cantidad de clases: " + str(
                clases) + " " + str(cantidad) + "\n"
        else:
            cadena += ("feature_" + str(node.feature) if not self.feature_names else self.feature_names[
                node.feature]) + " <= " + str(
                node.val_split) + "\n"
            cadena += self.text_tree(node.left, depth + 1)

            cadena += ("|" + " " * 3) * depth
            cadena += "|--- "
            cadena += ("feature_" + str(node.feature) if not self.feature_names else self.feature_names[
                node.feature]) + " > " + str(
                node.val_split) + "\n"
            cadena += self.text_tree(node.right, depth + 1)

        return cadena

    def export_tree(self):
        return self.text_tree(self.tree, 0)


if __name__ == '__main__':

    def encontrar_fila_con_palabra(ruta_archivo, palabra):
        with open(ruta_archivo, 'r') as archivo:
            for num_linea, linea in enumerate(archivo, 1):  # Empezamos desde la línea 1
                if palabra in linea:
                    return num_linea
        return -1


    def cross_val(name, p_unlabeled="20"):

        accuracy_ssl = []
        accuracy_dt = []
        accuracy_st = []

        print("PERCENTAGE:", p_unlabeled, "- DATASET:", name)
        for k in range(1, 11):
            train_data = pd.read_csv(
                f'datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tra.dat',
                header=None,
                skiprows=encontrar_fila_con_palabra(
                    f'datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tra.dat',
                    '@data'))

            test_data = pd.read_csv(
                f'datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tst.dat',
                header=None,
                skiprows=encontrar_fila_con_palabra(
                    f'datasets/{p_unlabeled}/{name}-ssl{p_unlabeled}-10-fold/{name}-ssl{p_unlabeled}/{name}-ssl{p_unlabeled}-10-{k}tst.dat',
                    '@data'))

            if pd.api.types.is_numeric_dtype(test_data.iloc[:, -1]):
                train_data.loc[train_data.iloc[:, -1] == ' unlabeled', len(train_data.columns) - 1] = -1
                train_data.iloc[:, -1] = pd.to_numeric(train_data.iloc[:, -1])
            else:
                label_encoder = LabelEncoder()
                # Codificar las etiquetas de clase
                train_data.iloc[:, -1] = label_encoder.fit_transform(train_data.iloc[:, -1])
                train_data.loc[train_data.iloc[:, -1] == label_encoder.transform([' unlabeled'])[0], len(
                    train_data.columns) - 1] = -1

                test_data.iloc[:, -1] = label_encoder.transform(test_data.iloc[:, -1])

            train_data[train_data.columns[-1]] = train_data[train_data.columns[-1]].astype(int)
            test_data[test_data.columns[-1]] = test_data[test_data.columns[-1]].astype(int)

            train_data_label = train_data[train_data.iloc[:, -1] != -1]

            my_tree = SSLTree(w=1)
            my_tree.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
            # print(my_tree.export_tree())
            # print(accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values)))

            dt = DecisionTreeClassifier()
            dt.fit(train_data_label.iloc[:, :-1].values, train_data_label.iloc[:, -1].values)
            # print(export_text(dt))
            # print(accuracy_score(test_data.iloc[:, -1].values, dt.predict(test_data.iloc[:, :-1].values)))

            self_training_model = SelfTrainingClassifier(DecisionTreeClassifier())
            self_training_model.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)

            accuracy_ssl.append(
                accuracy_score(test_data.iloc[:, -1].values, my_tree.predict(test_data.iloc[:, :-1].values)))
            accuracy_dt.append(accuracy_score(test_data.iloc[:, -1].values, dt.predict(test_data.iloc[:, :-1].values)))
            accuracy_st.append(accuracy_score(test_data.iloc[:, -1].values,
                                              self_training_model.predict(test_data.iloc[:, :-1].values)))
            print("\tFOLD", k, "- Done")

        return np.median(accuracy_ssl), np.median(accuracy_dt), np.median(accuracy_st)


    names = ["yeast", "iris", "appendicitis", "wine", "bupa", "dermatology", "glass", "sonar", "spectfheart", "vehicle",
             "vowel", "cleveland"]

    # Problemas con tae, thyroid, contraceptive

    all_medians = {}

    all_mean_rankings = np.empty((3, 4))

    for i, p in enumerate(["10", "20", "30", "40"]):
        medians_ssl = []
        medians_dt = []
        medians_st = []
        for name in names:
            m_ssl, m_dt, m_st = cross_val(name, p)
            # break
            medians_ssl.append(m_ssl)
            medians_dt.append(m_dt)
            medians_st.append(m_st)
        # break
        print(medians_ssl)
        print(medians_dt)
        print(medians_st)

        all_medians[p] = np.stack((medians_ssl, medians_dt, medians_st))

        rankings = rankdata(-all_medians[p], method="average", axis=0)
        print(rankings)

        all_mean_rankings[:, i] = np.mean(rankings, axis=1)

    final_rankings = rankdata(all_mean_rankings, method="average", axis=0)
    print(all_mean_rankings)

    plt.figure(figsize=(10, 6))

    for i, percentage in enumerate(["10%", "20%", "30%", "40%"]):
        top = all_mean_rankings[:, i].copy()
        uniques, _ = np.unique(top, return_counts=True)

        displacement = 0.05 * np.linspace(-1, 1, len(uniques), endpoint=False)
        dup = 0
        for j, value in enumerate(top):
            if np.count_nonzero(top == value) > 1:
                all_mean_rankings[j][i] += displacement[dup] if dup < len(displacement) else 0
                dup += 1

    classifiers = ["SSLTree", "DecisionTree", "SelfTraining"]
    for j, classifier in enumerate(classifiers):
        plt.scatter(["10%", "20%", "30%", "40%"], all_mean_rankings[j], label=classifier)

    plt.ylim(0, 3.5)
    plt.xlabel("Percentage")
    plt.ylabel("Ranking")
    plt.title("Comparativa SSLTree, DT y ST")

    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    # plt.scatter(medians_ssl, medians_dt, color='blue')
    plt.plot([min(medians_ssl + medians_dt) * 0.7, 1],
             [min(medians_ssl + medians_dt) * 0.7, 1], color='red', linestyle='--')

    colores = plt.cm.viridis(np.linspace(0, 1, len(medians_ssl)))

    # Agregar los puntos al gráfico de dispersión uno por uno
    for i in range(len(medians_ssl)):
        plt.scatter(medians_ssl[i], medians_dt[i], color=colores[i], label=names[i])

    plt.legend()
    plt.title("Median accuracy")
    plt.show()
