# K- Nearest Neighbors Classifier
- knn classifiers can classify unlabeled examples by assigning them the class of the most similar labeled examples
- simple and powerful
- knn is well suited for classification where relationship between features is complex/hard to understand
- use __Euclidean Distance__: `dist(x, y) = sqrt{(x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2}`
- k=1: consider smallest distance
- k=2: consider 2 smallest distances
- k=3: consider 3 smallest distances
- k=k: consider k smallest distances (i.e. consider all distances)
- k cannot be > number of features.
- if k is small then noisy data or outliers will have huge impact / underfitting
- if k is large then overfitting meaning majority class will be considered regardless of which neighbors are nearest

## Lazy Learner
- does not learn anything.
- just store training data -> slow prediction
- we don't build model
- __non-parametric learning__: non parameters are to be learned about the data (unlike with linear regression)
