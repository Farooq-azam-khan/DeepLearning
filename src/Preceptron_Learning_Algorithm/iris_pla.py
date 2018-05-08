from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from preceptron import Preceptron
from improved_preceptron import Improved_Preceptron

def get_iris_data(p_model):
    iris = load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()

    y_result = []
    X_result = []

    for target, x in zip(y, X):
        if p_model == "svi":
            if target == 0 or target == 2:
                # -1 because pla will output -1 or 1
                # and target is either 0, 1, 2 so will map to -1, and 1
                y_result.append(target-1)
                X_result.append(x)
        elif p_model == "sve":
            if target == 0:
                # 0-1 = -1 for that -1 mapping
                y_result.append(target-1)
                X_result.append(x)
            elif target == 1:
                # dont change for the 1 mapping
                y_result.append(target)
                X_result.append(x)
        elif p_model == "vive":
            if target == 2: #virginica
                y_result.append(-1)
                X_result.append(x)
            elif target == 1: #versicolor
                y_result.append(target)
                X_result.append(x)


    return X_result, y_result


def main():
    iris = load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()



    # print(X)
    # print(y)
    targets = []
    for indx, i in enumerate(y):
        if i == 0:
            targets.append([1,0,0])
        elif i ==1:
            targets.append([0,1,0])
        elif i==2:
            targets.append([0,0,1])
    # print(targets)
    # inputs: sepal_length, sepal_width, pedal_length, pedal_width
    # outpts: setosa, versicolor, verginica
    ip = Improved_Preceptron(4, 3)

    X_test, y_test, X_train, y_train = ip.train_test_split(X, targets)
    ip.fit(X_test, y_test, X_train, y_train)
    # for index, xi in enumerate(X):
    #     ip.train(xi, targets[index])

    print(ip.feed_forward(X[0]))
    print("expected:", targets[0])

    # three preceptron algorithms (because can only make 2 classifications)

    # 1. setosa, virginica (-1, 1) respectively
    # print("----setosa and virginica----")
    # p_setosa_virginica = Preceptron(4)
    # following data only contains irises with labels of setos and virginica
    # X_svi, y_svi = get_iris_data("svi")
    # svi_X_train, svi_y_train, svi_X_test, svi_y_test = p_setosa_virginica.train_test_split(X_svi, y_svi)
    # p_setosa_virginica.fit(svi_X_train, svi_y_train, svi_X_test, svi_y_test)

    # 2. setosa, versicolor (-1, 1) respectively
    # print("----setosa and versicolor----")
    # p_setosa_verisicolor = Preceptron(4)
    # X_sve, y_sve = get_iris_data("sve")
    # sve_X_train, sve_y_train, sve_X_test, sve_y_test = p_setosa_verisicolor.train_test_split(X_sve, y_sve)
    # p_setosa_verisicolor.fit(sve_X_train, sve_y_train, sve_X_test, sve_y_test)

    # 3. virginica, versicolor (-1, 1) respectively
    # print("----setosa and versicolor----")
    # p_virginica_versicolor = Preceptron(4)
    # X_vive, y_vive = get_iris_data("vive")
    # vive_X_train, vive_y_train, vive_X_test, vive_y_test = p_setosa_verisicolor.train_test_split(X_vive, y_vive)
    # p_setosa_verisicolor.fit(vive_X_train, vive_y_train, vive_X_test, vive_y_test)

    # although we are getting very good accuracy the three modes we developed are not that practical
    # so for this reason we need to delop a PLA that has 3 - outputs or n-outputs for any problem
    # see improved_preceptron.py for use
    # graph_sub_plots(sepal_length, sepal_width, petal_length, petal_width)


def graph_sub_plots(sepal_length, sepal_width, petal_length, petal_width):
    fig, axes = plt.subplots(2, 2) #, subplot_kw=dict(polar=True))

    fig.suptitle('Iris dataset')

    axes[0,0].scatter(sepal_length, sepal_width)
    axes[0,0].set_xlabel("sepal length")
    axes[0,0].set_ylabel("sepal width")

    axes[0,1].scatter(petal_length, petal_width)
    axes[0,1].set_xlabel("petal length")
    axes[0,1].set_ylabel("petal width")

    ax = fig.add_subplot(2,1,2,projection="3d")
    ax.scatter(sepal_length,petal_length,sepal_width)


    plt.show()



if __name__ == "__main__":
    main()
