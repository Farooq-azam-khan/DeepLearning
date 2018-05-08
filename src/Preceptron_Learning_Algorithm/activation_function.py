import math
'''
    class: ActivationFunction
    usage: contain collection of function that can be used as parameters
'''
class ActivationFunction():
    def __init__(self, func, dfunc):
        # function and its derivative
        self.func = func
        self.dfunc = dfunc

    def __repr__(self):
        return "f(x):{}, df(x)/dx:{}".format(self.func(x), self.dfunc(x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def dsigmoid(y):
    return y * (1-y)
def sign(x):
    if x>=0:
        return 1
    else:
        return -1


sigmoid_activation = ActivationFunction(sigmoid, dsigmoid)
sign_activation = ActivationFunction(sign, 0)
def main():
    print(sigmoid_activation.func(0), sigmoid_activation.dfunc(0))
if __name__ == "__main__":
    main()
