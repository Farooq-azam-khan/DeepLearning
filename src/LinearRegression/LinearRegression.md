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

  | __Design Matrices__ |   __Gradient Descent__    |
  |:--------------------|--------------------------:|
  | No parameters       |   Have to Choose LR       |
  | Matrix inversion!   |   No expensive Operations |
  | Lower dimensions    |   Higher Dimensions       |

## Gradient Descent
- graph `b0` and `b1`. Need to find minimum value of plane formed with graph. Need partial derivatives.
- The minimum value of `(b0, b1)` is the optimal one and thus will be values for our linear equation.

# Logistic Regression
- http://www.saedsayad.com/logistic_regression.htm
- Predicts the probability of an outcome that can only have two value (__dichotomy__).
- use __sigmoid__ function `p(x) = 1 / [1 + e^(-y)]` for `y = b0 + b1*x`
- gets a value between `0` and `1`.
- if we get non-linear curve, we can make it linear with __Logit Transformation__
  - `log(p / (1-p)) = b0 + b1x`
- how to fit the parameters?
  - __maximum likelihood method__
  - __gradient descent method__

## Multivariate Logistic Regression
- `p(x) = 1 / (1+e^[-(b0 + b1x1 + b2x2+...+bnxn)])`
- __sigmoid__ function is between 0 and 1 thus it is good for predicting probabilities
