import random
# TODO: implement unit testing
'''
    class: vector class
'''
class Vector():

    def __init__(self, dim):
        self.dim = dim # num of dimensions
        self.data = []
        self.initalize()

    def initalize(self):
        for _ in range(self.dim):
            self.data.append(0)

    def randomize(self):
        for indx, _ in enumerate(self.data):
            self.data[indx] = random.uniform(-1, 1)

    @staticmethod # could use __mult__
    def dot(vec1, vec2):
        if vec1.dim == vec2.dim:
            sum = 0
            for val1, val2 in zip(vec1.data, vec2.data):
                sum += (val1 * val2)
            return sum
        else:
            print("need to be same dimension")
            return None

    def __add__(self, other):
        if self.dim == other.dim:
            result = Vector(self.dim)
            for indx, data_self in enumerate(self.data):
                result.data[indx] = data_self + other.data[indx]
            return result
        else:
            print("must have same dimensions")
            return None

    def __sub__(self, other):
        if self.dim == other.dim:
            result = Vector(self.dim)
            for indx, data_self in enumerate(self.data):
                result.data[indx] = data_self - other.data[indx]
            return result
        else:
            print("must have same dimensions")
            return None

    def __repr__(self):
        ret = "["
        for indx, val in enumerate(self.data):
            if indx == len(self.data)-1:
                ret += "{:.2f}".format(val)
            else:
                ret += "{:.2f}, ".format(val)
        return ret + "]"

def main():
    vec1 = Vector(3)
    vec2 = Vector(3)
    vec1.randomize()
    vec2.randomize()
    print(vec1)
    print(vec2)
    print(Vector.dot(vec1, vec2))

if __name__=="__main__":
    main()
