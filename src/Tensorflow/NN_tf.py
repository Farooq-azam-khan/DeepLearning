from __future__ import absolute_import, division, print_function
import tensorflow as tf

# enable eager execution: https://www.tensorflow.org/programmers_guide/eager
tf.enable_eager_execution()

def ee():
    tf.executing_eagerly() # -> true
    x = [[2.0]]
    m = tf.matmul(x, x)
    print("m: {}".format(m))

    a = tf.constant([[1,2],
                    [3,4]])
    print(a)

    b = tf.add(a, 1)
    print("a+1:", b)

    print("a*b:", a*b)

    # use with numpy
    import numpy as np
    print("tensor objects can be used with numpy")
    c = np.multiply(a, b)
    print("c=a*b", c)

    # obtain numpy values from a tensor
    print("obtain numpy values from a tensor")
    print("a:", a.numpy())

def fizzbuzz():
    print(tf.executing_eagerly())
    # tf.enable_eager_execution()
    max_num = 20
    zero = tf.constant(0)
    counter = tf.constant(0)
    # print(counter)
    for num in range(max_num):
        num = tf.constant(num)
        div_3 = num%3
        div_5 = num%5
        # print(div_3==0)
        if div_3 == zero and div_5 == zero:
            print("FizzBuzz")
        elif div_3 == zero:
            print("Fizz")
        elif div_5 == zero:
            print("Buzz")

# building a layer with tf.keras.layers
class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        self.output_units = output_units
    # build method gets called first time the layer is used
    # creates variables on build() allows shapes to depend on input shape
    # removes need for user to specify shpae/ but still possible to build shape in __inti__()
    def build(self, input):
        self.kernel = self.add_variable("kernel", [input.shape[-1], self.output_units])

    def call(self, input):
        # override call() instead of __call__ we perform bookeeping.
        return tf.matmul(input, self.kernel)

def main():
    pass

if __name__ == "__main__":
