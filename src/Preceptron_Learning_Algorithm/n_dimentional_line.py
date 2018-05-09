from preceptron import Preceptron
import random

# does 3 d
# TODO: make it ndimentional
def main():

    # generate a 3d ponit with a label
    # point = Point()
    # generate a 3d ponit with a label
    points = [Point() for _ in range(10)]
    # print(points)

    prec_x_val = Preceptron(2)
    prec_y_val = Preceptron(2)
    prec_z_val = Preceptron(2)

    for _ in range(10):
        for _ in range(100):
            indx = random.randrange(len(points))
            point = points[indx]
            prec_x_val.train([point.t, point.r[0]], point.label[0])
            prec_y_val.train([point.t, point.r[1]], point.label[1])
            prec_z_val.train([point.t, point.r[2]], point.label[2])

    point = Point()
    fed = [prec_x_val.feed_forward([point.t, point.r[0]]),
    prec_y_val.feed_forward([point.t, point.r[1]]),
    prec_z_val.feed_forward([point.t, point.r[2]])]

    print("predicted:", fed)
    print("expected:",point.label)

def test_classes():
    point = Point_Nd(10)
    print(point.label)

''' generate an nd point with a label '''
class Point_Nd():
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.t = random.uniform(-1, 1)
        self.r = []
        # initalize point
        for i in range(self.dimension):
            self.r.append(random.uniform(-1, 1))

        # label
        self.label = []
        # initilize label with zeros
        for _ in range(self.dimension):
            self.label.append(0)

        # calculate the labe
        actual_line = Line_Nd(self.dimension)
        actual_r = actual_line.calc_r(self.t)
        # print("v:", actual_line.v)
        # print("r0:", actual_line.r0)
        # print("actual: r", actual_r)

        for i in range(self.dimension):
            if actual_r[i] > self.r[i]:
                self.label[i] = 1
            else:
                self.label[i] = -1

''' equation of an nd line '''
class Line_Nd():
    def __init__(self, dimension=3):
        self.dimension = dimension

        # r = r0 + vt
        self.v = []
        self.r0 = []


        # initalize r0 and v
        for i in range(self.dimension):
            self.r0.append(1)
            self.v.append(1)



    def calc_r(self, t):
        r = []
        for i in range(self.dimension):
            val = self.r0[i] + self.v[i] * t
            r.append(val)
        return r

        return [1,1,1]

    def __repr__(self):
        return "r = {} + {}t".format(self.r0, self.v)

''' equation of a 3d line '''
# function for the fomula of an n-dimentional line(we will work w/ 3d for graphing purposes)
# see for practice: http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfLines.aspx
# r = r_0 + tv
#      x, y, z
# r_0 = [1, 1, 1]
# v   = [1, 1, 1]
# r   = [r_0[0]+t*v[0], r_0[1]+t*v[1], r_0[2]+t*v[2]]
class Line3d():
    def __init__(self, t):
        self.t = t

        self.r0 = [1, 1, 1]
        self.v = [1, 1, 1]


        self.r = [0, 0, 0]
        self.calcR()

    def calcR(self):
        self.r[0] = self.r0[0] + self.v[0]*self.t
        self.r[1] = self.r0[1] + self.v[1]*self.t
        self.r[2] = self.r0[2] + self.v[2]*self.t

    def __repr__(self):
        return "{}".format(self.r)

class Point():
    def __init__(self):

        # generates a random t
        self.t = random.uniform(-5, 5)

        # generates random r
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(-10, 10)

        self.r = [x, y, z]

        # the label is the known answer. We use this to train preceptron.
        self.label = [0, 0, 0]

        actual_line = Line3d(self.t)
        #label the x
        if actual_line.r[0] < self.r[0]:
            self.label[0] = 1
        else:
            self.label[0] = -1

        #label the y
        if actual_line.r[1] < self.r[1]:
            self.label[1] = 1
        else:
            self.label[1] = -1

        # label the z
        if actual_line.r[2] < self.r[2]:
            self.label[2] = 1
        else:
            self.label[2] = -1
        # print("actual_r:", actual_line)
        # print("rand_r:", self.r)
        # print("label:", self.label)

    def __repr__(self):
        # format the output to look cleaner
        r = "[{:.2f}, {:.2f}, {:.2f}]".format(self.r[0], self.r[1], self.r[2])
        return "r:{}, label:{}".format(r, self.label)

if __name__ == "__main__":
    # main()
    test_classes()
