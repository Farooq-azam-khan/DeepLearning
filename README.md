# DeepLearning Library
Here is the tutorial for Deep Learning. This repository requires python 3
Follow these instruction so that you can use this repository:
1. Launch a python virtual environment by typing `virtualenv [env_name]`.
    - If you do not have virtuelenv use: `pip3 install virtualenv`.
2. Once it is setup activate the virtual environment by typing: `Scripts\activate`
    - This way the python moduels installed on your computer will not be affected.
3. Download the zip file and extract the `src\` file and the `requirements.txt` file in to the virtuelenv directory.
4. Install the required python packages from requirements.txt. This following code will install all of the requirements: `pip3 install -r requirements.txt`.
5. Now you can run any python file in this repository, just type: `python [file_name]` in that directory.

## 1. Preceptron Learning Algorithm
  - look at the following files: `preceptron.py`, `linear_function.py`, `boolean_function.py`.
  - the `Preceptron()` class contains the preceptron learning algorithm
  - the `linear_function.py` file contains a graphical understaning of how PLA does linear separation of data points.
    - there are four classification for a point. If the point is above the line then it is green and if it is below the line then it is red. If the algorithm predicts is accurately then it is marked as an `o` and if it predicts it incorrectly then it is an `x`; moreover, it also draws the correct line and predicted line.
  - the `boolean_function.py` file contains examples of PLA successes as well as its failure, ie. the XOR problem. Look at `neural_network.py` for improvement to PLA and a solution to the `xor` problem.
## 2. Neural Network Algorithm
  - In the directory `Neural_Network` you will find 2 files: `matrix.py` and `neural_network.py`.
  - the `matrix.py` contains matrix operations (which you can look at if you are interested but it is not necessary).
    - An interesting feature in this class is the `map(func)` method. If you are coming from `Java` it should be noted that, in `Python`, you can pass in functions to another function. That is precisely what is happening here. So if `func(x) = 2*x` then `map(func)` is allowed and will be `map(2*x)`.
  - `neural_network.py` contains two important functions, the `feed_forward(inputs)` and the `train(inputs, targets)` methods. Both expect arrays as parameters. Note that this is very similar to the Preceptron Learning Algorithm; however, the complexity to the algorithm comes from the linear algebra, and the calculus involved with it. The necessary linear algebra comes from the `matrix.py` file.
## 3. XOR problem
  - with the `neural_network.py` file as you can see the `xor` problem, although simple to us cannot be solved by the PLA but it is very easy for the NN (after 1000 iterations of training).
## 4. Tensorflow in python
  - in the director `Tensorflow` we have the following files: 'NN_tf.py', `iris_tf.py`
  - the `NN_tf.py` file contains an implementation of a Deep Neural Network with tensorflow. It trains a model on the mnist dataset which has 784 inputs, 3 hidden layers, and 10 outputs.
## 5. CNN and Keras

For you to use this repository you will need to install the requirements.
`pip install requirements.txt`
