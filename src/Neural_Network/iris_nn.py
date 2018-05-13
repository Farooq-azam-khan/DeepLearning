from sklearn.datasets import load_iris
from neural_network import NeuralNetwork
import random

# use of neural network on iris dataset
def get_iris_data():
    iris = load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()
    targets_by_name = []

    # create a one hot array with the targets
    for indx, target in enumerate(y):
        if target==0:
            y[indx] = [1, 0, 0]
            targets_by_name.append("setosa")
        elif target == 1:
            y[indx] = [0, 1, 0]
            targets_by_name.append("virginica")
        else:
            y[indx] = [0, 0, 1]
            targets_by_name.append("versicolor")
    return X, y, targets_by_name

def main():
    # 4 inputs for lengths, and 3 outputs for types of irises
    iris_nn = NeuralNetwork(4, 5, 3)
    inputs, targets, targets_by_name = get_iris_data()

    for _ in range(10):
        for _ in range(1000):
            indx = random.randrange(len(inputs))
            input = inputs[indx]
            target = targets[indx]
            iris_nn.train(input, target)

    # clean output
    print("{:5s} | {:20s} | {:10s} | {:10s}".format("Evalu", "Inputs", "Target", "Prediction"))
    print("{:45}".format("--------------------------------------------------"))
    for target, input in zip(targets_by_name, inputs):
        prediction = process_output(iris_nn.feed_forward(input))
        row = "{:5s} | {:20s} | {:10s} | {:10s}".format(str(target==prediction), str(input), target, prediction)
        print(row)

def process_output(array):
    max_indx = array.index(max(array))
    if max_indx == 0:
        return "setosa"
    elif max_indx == 1:
        return "virginica"
    elif max_indx == 2:
        return "versicolor"
    return "??"

if __name__ == "__main__":
    main()
