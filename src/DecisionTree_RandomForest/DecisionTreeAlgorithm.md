# Decision Tree
- Supervised learning technique for classification or regression
- Create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features
- not as effective, yet simple
- __Boosting__ and __Random Forest__ improves performance
- when making decisions one just has to traverse the tree based on the __features__
- good recourse: http://www.saedsayad.com/decision_tree_reg.htm
- Must look at the __Shannon Entropy__ (physics has the __Maxwell-Boltzmann entropy__)

## Advantages
- Simple to understand and to interpret + trees can be visualized (to be implemented)
- No need for data preparations such as normalization or dummy variables
- `O(log N)`

## Disadvantages
- Decision-tree learners can create over-complex trees that do not generalize the data well -> overfitting / underfitting
- Can be unstable because small variations in the data might result in a completely different tree
- optimal decision tree is NP-complete
- practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm
