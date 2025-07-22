# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

import torch
from sklearn.model_selection import train_test_split

n = 5
features = 2
X, y = torch.arange(n * features).reshape(n, features), range(n)

print(X)
# array([[0, 1],
#        [2, 3],
#        [4, 5],
#        [6, 7],
#        [8, 9]])

print(list(y))
# [0, 1, 2, 3, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


print("X_train:", X_train)
# array([[4, 5],
#        [0, 1],
#        [6, 7]])

print("y_train:", y_train)
# [2, 0, 3]

print("X_test:", X_test)
# array([[2, 3],
       # [8, 9]])

print("y_test:", y_test)
# [1, 4]
