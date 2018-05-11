# Machine Learning
- give computers ability to learn with data
- types: __Supervised__ and __Unsupervised Learning__

# Supervised
- Give algorithm the right answer so it can develop
- Example algorithms: __Regression, Naïve Based, SVM, Neural Nets__
- use a __dataset__
  - split it into __training__ and __testing__
  - __training set__ will train algorithm
  - __testing set__ will check performance

# Unsupervised
- Do not have a dataset with the right answers -> data w/out labels
- algorithm must find a pattern if it exists, called __clusters__
- Example algorithms: __K-means clustering, hierarchical clustering, PCA__

## Cross Validation (CV)
- Assess how the results of a statistical analysis will generalize to an independent dataset
- datasets are partitioned into __training set, test set, and validation set__
- __Exhaustive CV__: learn and test on all possible ways to divide test, train and validation set.
- __Non-exhaustive CV (k-fold CV)__: don't compute all ways of splitting original sample.

### K-Fold CV
- original sample is randomly partitioned into _k_ equal sized subsamples.
- single subsample is retained as validation data for testing model, and `k-1` subsamples are used as training.
- CV process in repeated `k` times w/ each of the `k` subsamples used exactly once as the validation data
- can be averaged for single estimation
- __Advantages__: all observations are used for both training and validation, and each observation is used for validation exactly once.

## Normalization
- features are usually transformed into a range before applying any sort of algorithm
- this is so that other features which may have large numbers are scaled equally as features with small number.
- important for KNN and neural networks because large distances will strongly dominate large values.
- two ways: __min-max normalization, z-transformation__

### Min-Max Normalization
- transforms a feature such that all values fall in range between 0 and 1
- `X_new = (x - min(x)) / (max(x) - min(x))`

### Z-transformation
- 'X_new = (x - mean(x)) / (standardDieviation(x))'
