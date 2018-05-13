import random # used to generate random weights
import math
from vector import Vector

# TODO: debug errors

# activation function
def sign(num):
    if (num >= 0):
        return 1
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y*(1-y)

'''
    class: Improved_Preceptron
    usage: same as preceptron class but here there are n outputs
 '''
class Improved_Preceptron():
    # constructor
    def __init__(self, num_inputs, num_outputs):
        self.num_weights = num_inputs
        self.num_outputs = num_outputs
        #learning rate
        self.lr = 0.1

        #len will be number of inputs
        self.input_weights = Vector(self.num_weights)
        # len will be number of outputs
        self.output_weights = Vector(self.num_outputs)

        self.input_weights.randomize()
        self.output_weights.randomize()
        self.input_bias = Vector(1)    # len will be 1
        self.output_bias = Vector(self.num_outputs)    # len will be number of biases

        self.input_bias.randomize()
        self.output_bias.randomize()

    def randmtd(self):
        return self.input_weights



    '''
        param: inputs array
        return: expected output
    '''
    def feed_forward(self, input_array):
        # feed_forward to preceptron
        inputs = Vector.to_vector(input_array)
        preceptron_summation = Vector.dot(inputs, self.input_weights)
        # add the bias
        preceptron_summation += self.input_bias.data[0] # array of 1 value
        # pass the sum through the activation function
        preceptron_summation = sigmoid(preceptron_summation)

        # feed_forward to output
        output = Vector(self.num_outputs)
        output = Vector.scalar_mul(self.output_weights, preceptron_summation)
        output = output + self.output_bias
        output.map(sigmoid)
        return output.to_array()

    '''
    param input: data you want to use to train preceptron
    param target: the known output for adjusting the weights ie the label
    '''
    def train(self, input_array, target_array):

        inputs = Vector.to_vector(input_array)
        targets = Vector.to_vector(target_array)

        # feed_forward to preceptron
        preceptron_summation = Vector.dot(inputs, self.input_weights)
        # add the bias
        preceptron_summation += self.input_bias.data[0] # array of 1 value
        # pass the sum through the activation function
        preceptron_summation = sigmoid(preceptron_summation)

        # feed_forward to output
        outputs = Vector(self.num_outputs)
        outputs = Vector.scalar_mul(self.output_weights, preceptron_summation)
        outputs = outputs + self.output_bias
        outputs.map(sigmoid)

        # --------- update output weights and biases ---------
        # output error
        output_errors = targets - outputs
        output_gradient = Vector.static_map(outputs, dsigmoid)
        delta_weight_output = output_errors * Vector.scalar_mul(output_gradient,preceptron_summation)
        delta_weight_output = Vector.scalar_mul(delta_weight_output, self.lr)

        self.output_weights = self.output_weights + delta_weight_output
        self.output_bias = self.output_bias + output_gradient

        # --------- update output weights and biases ---------
        # input error
        input_errors = Vector.dot(self.output_weights, output_errors)
        input_gradient = Vector.static_map(inputs, dsigmoid)
        delta_weight_input =  input_errors * Vector.dot(inputs, input_gradient)
        delta_weight_input = delta_weight_input * self.lr

        # adust weights and biases for input

        self.input_weights.scalar_add(delta_weight_input)
        self.input_bias.scalar_add(delta_weight_input)

    def __repr__(self):
        ret1 = "(input)  weights: {}, bias: {} \n".format(self.input_weights, self.input_bias)
        ret2 = "(output) weights: {}, biases: {}".format(self.output_weights, self.output_bias)
        return ret1+ret2


def main():
    p = Improved_Preceptron(2, 1)
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    labels = [[0], [1], [1], [1]]
    p.feed_forward([1, 1])
    for _ in range(100):
        rand_indx = random.randrange(len(inputs))
        input = inputs[rand_indx]
        label = labels[rand_indx]
        p.train(input, label)

    print("0 0", p.feed_forward([0,0]))
    print("0 1", p.feed_forward([0,1]))
    print("1 0", p.feed_forward([1,0]))
    print("1 1", p.feed_forward([1,1]))

if __name__ =="__main__":
    main()
