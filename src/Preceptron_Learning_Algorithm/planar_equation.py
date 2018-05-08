# from file import class
from preceptron import Preceptron
import random
# for graphing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



def plane(x, y):
    # z = m1x + m2y + b
    return 2*x + 2*y + 1


'''
    class: Point
    usage: Generates points which will be used to train preceptron
'''
class Point():
    def __init__(self):
        # generates random x and y inputs
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.z = random.uniform(-1, 1)

        # the label is the known answer. We use this to train preceptron.
        self.label = 0

        line_z = plane(self.x, self.y)
        if (self.z > line_z):
            self.label = 1
        else:
            self.label = -1

    def get_points(self):
        return [self.x, self.y, self.z]
    @staticmethod
    def split_points(points):
        percent_to_split = 0.6 # 60% will be training and 40% will be testing
        splitting_index = int(len(points)*percent_to_split)
        train_points = points[:splitting_index] # first 60% are training
        test_points = points[splitting_index:] # last 40% are testing
        return test_points, train_points

    # like the toString method in java
    def __repr__(self):
        return "[x: {:.2f}, y: {:.2f}, z:{:.2f}, label: {}]".format(self.x, self.y, self.z, self.label)

def main():
    model_3d = Preceptron(3)

    points = []
    num_points = 100
    for _ in range(num_points):
        points.append(Point())

    # split the data into training and testing data
    train_points, test_points = Point.split_points(points)

    EPOCHS = 10
    BATCH_SIZE = 300
    for epoch in range(EPOCHS):
        # training happens here
        for _ in range(BATCH_SIZE):
            rand_num_indx = random.randrange(len(train_points)) # choose a random index for points
            rand_train_point = train_points[rand_num_indx]
            # inputs array
            training_inputs = [rand_train_point.x, rand_train_point.y, rand_train_point.z]
            # where is when we adjust the weights by training the model
            model_3d.train(training_inputs, rand_train_point.label)

        accuracy = model_3d.accuracy(test_points)
        print("Epoch: {:2d} out of {:2d} accuracy: {:.2f}".format(epoch+1, EPOCHS, accuracy))

        if accuracy == 1.0:
            break

    graph_3d_points(points, model_3d)


def graph_3d_points(points, model_3d):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    correct_above_x = []
    correct_above_y = []

    correct_below_x = []
    correct_below_y = []

    wrong_above_x = []
    wrong_above_y = []

    wrong_below_x = []
    wrong_below_y = []

    for point in points:
        prediction = model_3d.feed_forward(point.get_points())
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
    ax.scatter(correct_above_x, correct_above_y, c='g', marker="o", label="correct prediction/above plane")
    ax.scatter(correct_below_x, correct_below_y, c='r', marker="o", label="correct prediction/below plane")
    ax.scatter(wrong_above_x, wrong_above_y, c='g', marker="x", label="wrong prediction/above plane")
    ax.scatter(wrong_below_x, wrong_below_y, c='r', marker="x", label="wrong prediction/below plane")

    px = [pt.x for pt in points]
    py = [pt.y for pt in points]
    px, py = np.meshgrid(np.array(px), np.array(py))
    pz = np.array([plane(ptx, pty) for ptx, pty in zip(px, py)])
    surf = ax.plot_wireframe(px, py, pz, rstride=2, cstride=2, alpha=0.2, color="k", label="Actual Plane")

    # draw the line that the preceptron think it it
    model3d_plane_x = np.array([pt.x for pt in points])
    model3d_plane_y = np.array([pt.y for pt in points])
    model3d_plane_x, model3d_plane_y = np.meshgrid(model3d_plane_x, model3d_plane_y)
    model3d_plane_z = np.array([model_3d.guess_z(x, y) for x, y in zip(model3d_plane_x, model3d_plane_y)])
    surf2 = ax.plot_wireframe(model3d_plane_x, model3d_plane_y, model3d_plane_z, rstride=2, cstride=2, alpha=0.2, color="y", label="PLA Plane")
    # plt.plot(p_line_x, p_line_y, color="y", label="PLA Line")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
