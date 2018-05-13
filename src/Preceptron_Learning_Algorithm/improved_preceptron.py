import random # used to generate random weights
import math

# TODO: debug overflow and algorithm
# activation function
def sign(num):
    if (num >= 0):
        return 1
    else:
        return -1

def sigmoid(x):
    x = round(x, 2)
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

        # preceptron keeps track of its own weights
        self.input_weights = []
        self.output_weights = []

        self.input_bias = []
        self.output_bias = []

        self.randomize_input_weights()
        self.randomize_output_weigths()
        self.randomize_input_bias()
        self.randomize_output_bias()




    '''
        param: inputs array
        return: expected output
    '''
    def predict(self, inputs):
        return self.feed_forward(inputs)

    def feed_forward(self, inputs):
        preceptron_summation = 0
        # loop through the weights
        for i, weight in enumerate(self.input_weights):
            # multiply the weights by the inputs at that index
            result = weight * inputs[i]
            # add it to total sum
            preceptron_summation = preceptron_summation + result
        # add the bias
        preceptron_summation += self.input_bias[0] # array of 1 value
        # pass the sum through the activation function
        preceptron_summation = round(preceptron_summation, 30)
        preceptron_summation = sigmoid(preceptron_summation)

        # more than one output
        output = []
        for indx, o_weight in enumerate(self.output_weights):
            # multiply the weights and add the bias (bias is array of n elements)
            get_output = (preceptron_summation*o_weight)+self.output_bias[indx]
            # pass through activation function
            get_output = round(get_output, 30)
            output.append(sigmoid(get_output))

        return output

    '''
    param input: data you want to use to train preceptron
    param target: the known output for adjusting the weights ie the label
    '''
    def train(self, inputs, targets):
        preceptron_summation = 0
        # loop through the weights
        for indx, weight in enumerate(self.input_weights):
            preceptron_summation += weight * inputs[indx]

        # add the bias
        preceptron_summation += self.input_bias[0]
        # pass the sum through the activation function
        preceptron_summation = round(preceptron_summation, 30)
        preceptron_summation = sigmoid(preceptron_summation)

        guesses = []
        for indx, o_weight in enumerate(self.output_weights):
            # multiply the weights and add the bias (bias is array of n elements)
            get_output = (preceptron_summation*o_weight)+self.output_bias[indx]
            get_output = round(get_output, 30)
            activation = sigmoid(get_output)
            # pass through activation function
            guesses.append(activation)

        # get the error = known answer - guess
        output_errors = []
        for target, guess in zip(targets, guesses):
            output_errors.append(target - guess)

        # adjust the output weights here
        for indx, o_weight in enumerate(self.output_weights):
            output_gradient = dsigmoid(o_weight)
            output_error = output_errors[indx]
            delta_o_weight = self.lr * output_error * output_gradient * preceptron_summation
            self.output_weights[indx] = o_weight + delta_o_weight
            self.output_bias[indx] += output_gradient

        # ------ Adjust the input weights and biases here ------
        # find the error
        input_errors = [] # will only be one error because only one node
        input_error_sum = 0
        for indx, output_error in enumerate(output_errors):
            input_error_sum += self.output_weights[indx]
        input_errors.append(input_error_sum)
        # find the gradient
        input_gradient = self.lr * input_errors[0]

        for indx, input_weight in enumerate(self.input_weights):
            delta_i_weights = self.lr * input_errors[0] * input_gradient * inputs[indx]
            self.input_weights[indx] += delta_i_weights

        # adjsut the input biases
        self.input_bias[0] += input_gradient


    '''
        runs train function over and over agian
    '''
    def fit(self, inputs_train_array, targets_train_array, inputs_test_array, targets_test_array):
        EPOCHS = 10
        BATCHSIZE = 100
        for epoch in range(EPOCHS):
            for _ in range(BATCHSIZE):
                random_index = random.randrange(len(inputs_train_array))
                input = inputs_train_array[random_index]
                target = targets_train_array[random_index]
                self.train(input, target)
            acc = self.actual_accuracy(inputs_test_array, targets_test_array)
            print("Epoch: {} out of {} accuarcy: {:.2f}".format(epoch+1, EPOCHS, acc))


            if acc == 1.0:
                break

    def actual_accuracy(self, inputs_test_array, targets_test_array):
        average = 0
        # loop through the points
        for x, y in zip(inputs_test_array, targets_test_array):
            # get the inputs
            prediction = self.feed_forward(x)
            if prediction == y:
                average+=1

        return average / len(inputs_test_array)

    def train_test_split(self, X, y, train_split=0.5):
        # this shuffles data keeping the mapping of the two lists in check
        '''
        src: https://stackoverflow.com/questions/13343347/randomizing-two-lists-and-maintaining-order-in-python
        a = ["Spears", "Adele", "NDubz", "Nicole", "Cristina"]
        b = [1, 2, 3, 4, 5]
        combined = list(zip(a, b))
        random.shuffle(combined)
        a[:], b[:] = zip(*combined)
        '''

        combined = list(zip(X, y))
        random.shuffle(combined)
        X[:], y[:] = zip(*combined)


        splitting_index_X = int(len(X)*train_split)
        splitting_index_y = int(len(y)*train_split)

        X_train = X[:splitting_index_X]
        y_train = y[:splitting_index_y]
        X_test  = X[splitting_index_X:]
        y_test  = y[splitting_index_y:]
        return X_train, y_train, X_test, y_test

    def randomize_input_weights(self):
        for weight_index in range(self.num_weights):
            # initalize the weights to be random number between -1 and 1
            self.input_weights.append(random.uniform(-1,1))
    def randomize_output_weigths(self):
        for weight_index in range(self.num_outputs):
            # initalize the weights to be random number between -1 and 1
            self.output_weights.append(random.uniform(-1,1))
    def randomize_input_bias(self):
        for _ in range(1): # only one bias for the inputs
            self.input_bias.append(random.uniform(-1,1))
    def randomize_output_bias(self):
        for _ in range(self.num_outputs):
            self.output_bias.append(random.uniform(-1,1))

    '''
        param: array of points objects
        return: accuracy between 0 and 1
    '''
    def accuracy(self, points):

        average = 0
        # loop through the points
        for point in points:
            # get the inputs
            inputs = point.get_points()
            prediction = self.feed_forward(inputs)
            if prediction == point.label:
                average+=1

        return average / len(points)

    '''
        param: point for which you want to know the y value
        usage: get the y value of the linear function
    '''
    def guess_y(self, x):
        # w0*x + w1*y + w2*b
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]

        m = -(w0/w1)
        b = -(w2/w1) # * 1
        y = m * x + b
        return y

    '''
        param: point (x,y) for which you want to know the z value
        usage: get the z value of the plane
    '''
    def guess_z(self, x, y):
        # w0x + w1y + w2z + w3b = 0
        if len(self.weights) == 4:
            w0 = self.weights[0]
            w1 = self.weights[1]
            w2 = self.weights[2]
            w3 = self.weights[3]

            m1 = -(w0/w2)
            m2 = -(w1/w2)
            b = -(w3/w2) # *1
            z = m1*x + m2*y + b
            return z


def main():
    or_preceptron = Improved_Preceptron(2, 1)
    #     input 1, input 2, bias
    val0 = [0,0]
    val1 = [0,1]
    val2 = [1,0]
    val3 = [1,1]
    val_labels = [[0], [1], [1], [1]]
    inputs = [val0, val1, val2, val3]

    for _ in range(100):
        rand_num = random.randrange(0, 4)
        input = inputs[rand_num]
        label = val_labels[rand_num]
        or_preceptron.train(input, label)

    print("x or y")
    for i in inputs:
        # return -1 or 1
        pred = or_preceptron.feed_forward(i)
        # print("pred", pred)
        if pred == -1:
            pred = 0
        else:
            pred = 1
        print("{} | {} -> {}".format(i[0], i[1], pred))


if __name__ =="__main__":
    main()
