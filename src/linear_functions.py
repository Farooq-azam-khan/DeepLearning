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
        self.bias = 1 # necessary for y = mx + b

        # the label is the known answer. We use this to train preceptron.
        self.label = 0

        # actual y value for that particular line
        line_y = line(self.x)

        # latel is 1 if self.y is above the line
        if(self.y > line_y):
            self.label = 1
        else:
            self.label = -1

    # like the toString method in java
    def __repr__(self):
        return "[x: {:.2f}, y: {:.2f}, label: {}]".format(self.x, self.y, self.label)

def main():
    p = Preceptron(3)
    # test input
    inputs = [1, -1, 1]
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
        training_inputs = [point.x, point.y, point.bias]
        p.train(training_inputs, point.label)

        # dont train further if accuracy is perfect
        if p.accuracy(points) == 1.0:
            break;

        # show graph every thirty points
        # if i%30 == 0:
        #     showpoints(points, p)

    # now lets see what we get when we predict
    print("accuracy:", p.accuracy(points))
    print("output after training:", p.feed_forward(inputs))

    # graph a scatter plot of the data
    for point in points:
        prediction = p.feed_forward([point.x, point.y, point.bias])
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

    # draw the line that the preceptron think it it
    p_line_x = [i for i in range(-50, 50)]
    p_line_y = [p.guess_y(i) for i in p_line_x]
    plt.plot(p_line_x, p_line_y, color="y")
    # graph the line that the preceptron need to be trained towards
    line_x = [i for i in range(-50, 50)] # one line for loop
    line_y = [line(i) for i in line_x] # one line for loop
    plt.plot(line_x, line_y, color="k")
    plt.show()

if __name__ == "__main__":
    main()
