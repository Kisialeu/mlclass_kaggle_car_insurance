import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from load_car_insurance_with_region import load_train_and_test


# read data
train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv",
                                         "../data/car_insurance_test.csv")

# params for grid search
params = {'n_neighbors': list(range(3, 12))}

# make an instance of a grid searcher
best_clf = GridSearchCV(KNeighborsClassifier(), params, verbose=True, n_jobs=4,
                        scoring="roc_auc")

# fit X and y (train set and corresponding labels) to the grid searcher
best_clf.fit(train_df, y)

# print best estimatior and params
print("Best params:", best_clf.best_params_)
print("Best cross validation ROC AUC score", best_clf.best_score_)

# make predictions. This results in 0.764 AUC score
predicted_labels = best_clf.predict(test_df)

# turn predictions into data frame and save as csv file
predicted_df = pd.DataFrame(predicted_labels,
                            index = np.arange(1, test_df.shape[0] + 1),
                            columns=["too_much"])
predicted_df.to_csv("../output/knn_car_insurance.csv", index_label="id")

# that's for those who know the answers :)
# expected_labels_df = pd.read_csv("../data/car_insurance_test_labels.csv",
#                                  header=0, index_col=0)
# expected_labels = expected_labels_df['too_much']
# print(roc_auc_score(predicted_labels, expected_labels))

