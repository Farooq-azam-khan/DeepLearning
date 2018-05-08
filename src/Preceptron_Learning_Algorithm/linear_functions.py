import random
from preceptron import Preceptron
import matplotlib.pyplot as plt

# function for the fomula of the line we want preceptron to get
def line(x):
    # y = mx + b
    return -0.3 * x + 2

'''
    class: Point
    usage: Generates points which will be used to train preceptron
'''
class Point():
    def __init__(self):
        # generates random x and y inputs
        self.x = random.uniform(-50, 50)
        self.y = random.uniform(-50, 50)

        # the label is the known answer. We use this to train preceptron.
        self.label = 0

        # actual y value for that particular line
        line_y = line(self.x)

        # latel is 1 if self.y is above the line
        if(self.y > line_y):
            self.label = 1
        else:
            self.label = -1

    def get_points(self):
        return [self.x, self.y]

    # like the toString method in java
    def __repr__(self):
        return "[x: {:.2f}, y: {:.2f}, label: {}]".format(self.x, self.y, self.label)

    @staticmethod
    def split_points(points):
        percent_to_split = 0.6 # 60% will be training and 40% will be testing
        splitting_index = int(len(points)*percent_to_split)
        test_points = points[:splitting_index] # first 60% are training
        train_points = points[splitting_index:] # last 40% are testing
        return test_points, train_points

def main():
    #               number of inputs
    p_model = Preceptron(2)

    # here is our training data (just some random points generated with a label)
    points = []
    num_points = 100
    for _ in range(num_points):
        points.append(Point())

    # split the data into training and testing data
    train_points, test_points = Point.split_points(points)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        # training happens here
        for _ in range(200):
            rand_num_indx = random.randrange(0, len(train_points)) # choose a random index for points
            rand_train_point = train_points[rand_num_indx]
            # inputs array
            training_inputs = [rand_train_point.x, rand_train_point.y]
            # where is when we adjust the weights by training the model
            p_model.train(training_inputs, rand_train_point.label)

        accuracy = p_model.accuracy(test_points)
        print("Epoch:", epoch+1, "out of", EPOCHS, "accuracy:", accuracy)

        # dont train further if accuracy is perfect
        if accuracy == 1.0:
            break;

    # graph the lines and points of the models
    graph_model(points, p_model)


def graph_model(points, p_model):
    # graph a scatter plot of the data

    correct_above_x = []
    correct_above_y = []

    correct_below_x = []
    correct_below_y = []

    wrong_above_x = []
    wrong_above_y = []

    wrong_below_x = []
    wrong_below_y = []
    # classify points into four groups
    for point in points:
        prediction = p_model.feed_forward(point.get_points())
        # correct prediction and above line (green and circle)
        if prediction == point.label and point.label==1:
            correct_above_x.append(point.x)
            correct_above_y.append(point.y)
        # correct prediction and below line (red and circle)
        elif prediction == point.label and point.label==-1:
            correct_below_x.append(point.x)
            correct_below_y.append(point.y)
        # wrong prediction and above line (green and cross)
        elif prediction != point.label and point.label ==1:
            wrong_above_x.append(point.x)
            wrong_above_y.append(point.y)
        # wrong prediction and below line (red and cross)
        elif prediction != point.label and point.label ==-1:
            wrong_below_x.append(point.x)
            wrong_below_y.append(point.y)

    # draw the points
    plt.scatter(correct_above_x, correct_above_y, c='g', marker="o", label="correct prediction/above line")
    plt.scatter(correct_below_x, correct_below_y, c='r', marker="o", label="correct prediction/below line")
    plt.scatter(wrong_above_x, wrong_above_y, c='g', marker="x", label="wrong prediction/above line")
    plt.scatter(wrong_below_x, wrong_below_y, c='r', marker="x", label="wrong prediction/below line")

    # draw the line that the preceptron think it it
    p_line_x = [i for i in range(-50, 50)] # one line for loop
    p_line_y = [p_model.guess_y(i) for i in p_line_x]
    plt.plot(p_line_x, p_line_y, color="y", label="PLA Line")
    # graph the line that the preceptron need to be trained towards

    line_x = [i for i in range(-50, 50)]
    line_y = [line(i) for i in line_x]
    # display a legend to the graph
    plt.plot(line_x, line_y, color="k", label="Actual Line")

    # Place a legend to the right of this smaller subplot.
    plt.legend()
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.show()

if __name__ == "__main__":
    main()
