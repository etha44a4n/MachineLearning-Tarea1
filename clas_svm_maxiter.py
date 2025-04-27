import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo

print("Starting script")
#Import dataset
wine_quality = fetch_ucirepo(id=186)

#Transform to pandas df
wq_df = pd.DataFrame(data=wine_quality.data.features, columns=wine_quality.data.feature_names)
wq_df['quality'] = wine_quality.data.targets

#Generate synthetic data up to 10000 total rows
print("Generating synthetic data")
shuffled_df = shuffle(wq_df)

synthetic_data = shuffled_df.iloc[:3503].copy()

for column in synthetic_data.select_dtypes(include=np.number):
    if column == 'quality':
        continue
    synthetic_data[column] += np.random.normal(0, 0.1, len(synthetic_data))

combined_data = pd.concat([wq_df, synthetic_data], ignore_index=True)
print("Synthetic data generated")

#Separate train & test
X = combined_data.drop('quality', axis=1)
y = combined_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define parameter grid for hyperparameter optimization
print("Starting hyperparameter tuning (This could take a while)")
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100], #Regularization
    'gamma': ['scale', 'auto', 0.1, 1], #kernel coefficient for rbf, poly & sigmoid
    'random_state': [42],
    'max_iter': [50]
}

svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Hyperparameter tuning finished")

best_params = grid_search.best_params_
best_svc = grid_search.best_estimator_

print("Best hyperparameters", best_params)

y_pred_best = best_svc.predict(X_test)

#Print metrics
print(f"MSE: {mean_squared_error(y_test, y_pred_best)}")
print(f"MSE**0.5: {mean_squared_error(y_test, y_pred_best)**0.5}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_best)}")
print(f"r2_score: {r2_score(y_test, y_pred_best)}")

#Plot predictions against real results
plt.scatter(y_test, y_pred_best)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Best SVC Model Prediction')
plt.show()
