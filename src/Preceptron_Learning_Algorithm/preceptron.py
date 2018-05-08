import random # used to generate random weights

# activation function
def sign(num):
    if (num >= 0):
        return 1
    else:
        return -1

'''
    class: Preceptron
    usage: create a linearly separable line based on weights and biases
 '''
class Preceptron():
    # constructor
    def __init__(self, num_weights):
        # weights: the number of weights = number of inputs
        self.num_weights = num_weights
        # preceptron keeps track of its own weights
        self.weights = []
        for weight_index in range(self.num_weights):
            # initalize the weights to be random number between -1 and 1
            self.weights.append(random.uniform(-1,1))

        # add a random weight for the bias
        self.weights.append(random.uniform(-1,1))
        self.num_weights += 1 # this is becuase we added the bias

        #learning rate
        self.lr = 0.1

    '''
        param: inputs array
        return: expected output
    '''
    def feed_forward(self, inputs):

        # add the bias as part of the input
        bias = 1
        inputs.append(bias)

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

    '''
        param: point for which you want to know the y value
        usage: get the y value of the linear function
    '''
    def guess_y(self, x):
        # w0*x0 + w1*y + w2*b
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]
        m = -(w0/w1)
        b = -(w2/w1) # * 1
        y = m * x + b
        return y
