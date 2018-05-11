# Linear Regression
- Ask the question: is there a relation ship between two sets of data?
- if there is then let us assume it is linear: `y = mx + b` or `y = b0 + b1 * x`
- `x` will be the __independent variable / feature__ `y` will be the __dependent variable / label__
- simple/popular model in practice
- aim is to find `H(x) = b0 + b1*x` i.e. find `b0`, and `b1`.
  - Two main approaches to finding them: __design matrices__ and __gradient descent__

- __Least Square Method__: minimize the distance between the points `(x, y)` with our ideal line `H(x)`.
  - this means minimize: `[H(x) - y]^2`, where `H(x)` is algorithm's prediction and `y` is ideal line.
  - can use __gradient descent__ in higher dimensions (very efficient) or w/ __design matrices__ if there are not much features

  | __design matrices__ |   __gradient descent__    |
  |:--------------------|--------------------------:|
  | no parameters       |   have to choose LR       |
  | Matrix inversion!   |   No expensive operations |
  | Lower dimensions    |   higher dimensions       |
