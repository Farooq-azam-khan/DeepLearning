# DeepLearning Library
Here is the tutorial for Deep Learning. This repository requires python 3
Follow these instruction so that you can use this repository:
1. Launch a python virtual environment by typing `virtualenv [env_name]`.
    - If you do not have virtualenv use: `pip3 install virtualenv`.
2. Once it is setup activate the virtual environment by typing: `Scripts\activate`
    - This way the python moduels installed on your computer will not affect your moduels in this virtual environmnet nor will they be affected.
3. Download the zip file and extract the `src\` file and the `requirements.txt` file in to the virtuelenv directory.
4. Install the required python packages from requirements.txt. This following code will install all of the requirements: `pip3 install -r requirements.txt`.
    - To check that you have done the above steps correctly just type `pip freeze` to see all the packages that are installed on your virtual environment.
5. Now, you can run any python file in this repository, just type `python [file_name].py` in that directory.

## 1. Perceptron Learning Algorithm (PLA)
  - To learn about PLA look at the following files: `preceptron.py`, `linear_function.py`, `boolean_function.py`, and `planar_equation.py`.
  - In the `preceptron.py` file, the `Perceptron()` class contains the perceptron learning algorithm. This is an extremely useful algorithm to understand because neural networks and deep neural networks build on this simple algorithm.
    - The algorithm itself has two main parts to it, predicting results based on input and training based on desired outcome and actual outcome.
    - The reason PLA is simple and not useful in modern day research is because it can only predict linearly separable data (i.e. it can only separate things with a line and can be proven with linear algebra).
  - The `linear_function.py` file contains a graphical understanding of how PLA does linear separation of 2d inputs.
    - The `Point()` class generates random points with a label. The `label=1` if it is above the actual line and `label=-1` if it is below. PLA will try to approximate this line as best as possible by putting a point to either side of a line. Think of it as organizing a bowl of dimes and nickels. The dimes will go in one basket and the nickels in another.
    - There are four outcomes when predicting a set of inputs. It could be false positive, false negative, true positive, and true negative. These can be seen in the legend of the graph.
  - The `boolean_function.py` file contains examples of PLA successes as well as its failure, i.e. the XOR problem. Look at `neural_network.py` for improvement to PLA and a solution to the `xor` problem.
## 2. Neural Network Algorithm
  - In the directory `Neural_Network` you will find 2 files: `matrix.py` and `neural_network.py`.
  - the `matrix.py` contains matrix operations (which you can look at if you are interested but it is not necessary for an intuitive understanding).
  - `neural_network.py` contains two important functions, the `feed_forward(inputs)` and the `train(inputs, targets)` methods. Both expect arrays as parameters. Note that this is very similar to the Perceptron Learning Algorithm; however, the complexity to the algorithm comes from the linear algebra, and the calculus involved with it. This is because we are storing out weights and biases in a matrix and in some cases trying to get the derivative/gradient of that matrix. The necessary linear algebra comes from the `matrix.py` file.
    - An interesting feature in this class is the `map(func)` method. If you are coming from `Java` it should be noted that, in `Python`, you can pass in functions to another function, i.e. a function can be treated as a parameter. For example, if `func(x) = 2*x` then `map(func)` is allowed and will be `map(2*x)`.
    - On the subject of python, there are no such thing as `array` or `ArrayList`. This is just `list` which behaves like an `ArrayList`.
    - You do not have to worry about double or single quotation you can use either as long as you are consistent.
    - Common syntax for writing a `list` is by using a `for each loop`, which in python is the default `for loop`.

```python
letters = ['a', 'b', "c"]
# letter will equal to 'a', then 'b', adn then 'c'
for letter in letters:
  print(letter)
# enumerate() function will allow you to access the index of a function
for index, letter in enumerate(letters):
  print("index: {} -> letter: {}".format(index, letter))

# you can also write a list in one line
# here range() function return a list of numbers from 0 up to 9
# the for loop iterates over the list and gets the value at that index and stores
# it in number variable which is then added to the numbers list
numbers = [number for number in range(0, 10)]
print(numbers)

# above code does same thing as below
numbers2 = []
for number in range(10):
  numbers2.append(number)
print(numbers2)

# here is some practice: try to create a 2-d array using one line for loop
```
## 3. XOR problem
  - with the `neural_network.py` file as you can see the `xor` problem, although simple to us cannot be solved by the PLA but it is very easy for the NN (after 1000 iterations of training).
## 4. Tensorflow in python
  - in the director `Tensorflow` we have the following files: 'NN_tf.py', `iris_tf.py`
  - the `NN_tf.py` file contains an implementation of a Deep Neural Network with tensorflow. It trains a model on the mnist dataset which has 784 inputs, 3 hidden layers, and 10 outputs. The mnist dataset is data that contains hand written digits. With out model we are trying to predict which digits it is hence the 10 outputs (for the 10 digits) and the 784 is each pixel in the image. 
## 5. CNN and Keras
  - to be implemented
