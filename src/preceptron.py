import random
import matplotlib.pyplot as plt
# activation function
def sign(num):
    if (num >= 0):
        return 1
    else:
        return -1

''' Generates points which will be used to train preceptron '''
class Point():
    def __init__(self):
        # generates random x and y inputs
        self.x = random.uniform(0, 100)
        self.y = random.uniform(0, 100)

        # the label is the known answer. We use this to train preceptron
        self.label = 0

        # line y = x
        if (self.y > self.x):
            self.label = 1
        else:
            self.label = -1

    # like the toString method in java
    def __repr__(self):
        return "[x: {:.2f}, y: {:.2f}, label: {}]".format(self.x, self.y, self.label)



# class based example
class Preceptron():

    # constructor
    def __init__(self):
        # weights: the number of weights = number of inputs
        # preceptron keeps track of its own weights
        self.weights = []
        self.num_weights = 2
        for weight_index in range(0, self.num_weights):
            # initalize the weights to be random number between -1 and 1
            self.weights.append(random.uniform(-1,1))

        #learning rate
        self.lr = 0.1

    '''
        param: inputs array
        return: expected output
    '''
    def feed_forward(self, inputs):

        sum = 0
        # loop through the weights
        for i, weight in enumerate(self.weights):
            # multiply the weights by the inputs at that index
            result = weight * inputs[i]
            # add it to total sum
            sum = sum + result
        # pass the sum through the activation function
        output = sign(sum)
        # return the output
        return output

    '''
    param input: data you want to use to train preceptron
    param target: the known output for adjusting the weights ie the label
    '''
    def train(self, inputs, target):
        # get a guess based on the input (+1 or -1)
        guess_inputs = self.feed_forward(inputs)

        # get the error = known answer - guess
        error = target - guess_inputs

        # adjust the weights here
        for indx, weight in enumerate(self.weights):
            # change the weights based on the previous weight, the LR,
            # the error and its corresponding input
            delta_weight = error * inputs[indx] * self.lr
            self.weights[indx] = weight + delta_weight

    '''
        param: array of points objects
        return: accuracy between 0 and 1
    '''
    def accuracy(self, points):

        average = 0
        # loop through the points
        for point in points:
            # get the inputs
            inputs = [point.x, point.y]
            prediction = self.feed_forward(inputs)
            if prediction == point.label:
                average+=1

        return average / len(points)

def and_prediction():
    and_preceptron = Preceptron()
    val0 = [0,0]
    val1 = [0,1]
    val2 = [1,0]
    val3 = [1,1]
    val_labels = [0, 0, 0, 1]
    inputs = [val0, val1, val2, val3]

    for _ in range(1000):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        # expects target to be -1 or 1
        if label == 0:
            label = -1
        else:
            labe = 1
        # print('randnum:',rand_num, 'input:', input, 'label:', label)
        and_preceptron.train(input, label)

    for i in inputs:
        # return -1 or 1
        pred = and_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} | {} -> {}".format(i[0], i[1], pred))

def showpoints(points, p):
    for point in points:
        prediction = p.feed_forward([point.x, point.y])
        # correct prediction and above line (green and circle)
        if prediction == point.label and point.label==1:
            plt.scatter(point.x, point.y, c='g', marker="o")
        # correct prediction and below line (red and circle)
        elif prediction == point.label and point.label==-1:
            plt.scatter(point.x, point.y, c='r', marker="o")
        # wrong prediction and above line (green and cross)
        elif prediction != point.label and point.label ==1:
            plt.scatter(point.x, point.y, c='g', marker="x")
        # wrong prediction and below line (red and cross)
        elif prediction != point.label and point.label ==-1:
            plt.scatter(point.x, point.y, c='r', marker="x")

    # graph the line that the preceptron need to be trained towards
    line_x = [i for i in range(0, 100)] # one line for loop
    line_y = [i for i in range(0, 100)] # one line for loop
    plt.plot(line_x, line_y)

    plt.show()



def main():
    p = Preceptron()
    # test input
    inputs = [1, -1]
    output = p.feed_forward(inputs)
    print("inputs:", inputs)
    print("output for random weights:", output)

    # here is our training data
    points = []
    num_points = 50
    for _ in range(0, num_points):
        points.append(Point())
    # print("points used for training:", points)

    # training happens here
    # p.train([points[0].x, points[1].y], points[1].label
    for i, point in enumerate(points):
        # inputs array
        training_inputs = [point.x, point.y]
        p.train(training_inputs, point.label)

        # dont train further if accuracy is perfect
        if p.accuracy(points) == 1.0:
            break;

        # show graph every thirty points
        if i%30 == 0:
            showpoints(points, p)

    # now lets see what we get when we predict
    print("accuracy:", p.accuracy(points))
    print("output after training:", p.feed_forward(inputs))

    # graph a scatter plot of the data
    for point in points:
        prediction = p.feed_forward([point.x, point.y])
        # correct prediction and above line (green and circle)
        if prediction == point.label and point.label==1:
            plt.scatter(point.x, point.y, c='g', marker="o")
        # correct prediction and below line (red and circle)
        elif prediction == point.label and point.label==-1:
            plt.scatter(point.x, point.y, c='r', marker="o")
        # wrong prediction and above line (green and cross)
        elif prediction != point.label and point.label ==1:
            plt.scatter(point.x, point.y, c='g', marker="x")
        # wrong prediction and below line (red and cross)
        elif prediction != point.label and point.label ==-1:
            plt.scatter(point.x, point.y, c='r', marker="x")

    # graph the line that the preceptron need to be trained towards
    line_x = [i for i in range(0, 100)] # one line for loop
    line_y = [i for i in range(0, 100)] # one line for loop
    plt.plot(line_x, line_y)
    plt.show()

if __name__ == "__main__":
    main()
    # and_prediction()
