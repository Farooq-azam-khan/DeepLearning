# Naïve Bayesian Classifier
- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- Efficient supervised learning method even in higher dimensions
- can compete w/ SVM and random forests
- make good predictions with small data
1. need a __prior probability__: probability of each class in training set
2. get a circle w/ given radius and find number of data points in that circle call it `v`. The total number of a data point in a training set is `V`.
3. also find the probability of each class in that circle, `p'(v | V)`. Do it for every class.
4. __posterior probability__: __prior prob__ * `p'` for each class.
5. max of the __posterior probability__ is the class of that sample point.
