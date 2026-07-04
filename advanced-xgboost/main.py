from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import numpy as np

X,y=make_classification(n_samples=1000,n_features=20,n_informative=10,n_redundant=5,n_classes=2,random_state=42)

xgb_model=XGBClassifier(objective='binary:logistic',eval_metric="logloss",use_label_encoder=False,random_state=42)

param_grid={
    'n_estimators':[100,200],
    'learning_rate':[0.05,0.1],
    'max_depth':[3,5],
    'subsample':[0.7,0.9],
    'colsample_bytree':[0.7,0.9],
    'gamma':[0,0.1],
    'reg_alpha':[0,0.005]
}

param_distributions={
    'n_estimators':[100,200,300],
    'learning_rate':np.linspace(0.01,0.2,5),
    'max_depth':[3,5,7],
    'subsample':np.linspace(0.6,1.0,5),
    'col_sample_bytree':np.linspace(0.6,1.0,5),
    'gamma':[0,0.1,0.2,0.3],
    'reg_alpha':[0,0.001,0.005,0.01,0.05]
}

grid_search=GridSearchCV(estimator=xgb_model,param_grid=param_grid,scoring='accuracy',cv=3,verbose=1,n_jobs=-1)
grid_search.fit(X,y)
rand_search=RandomizedSearchCV(estimator=xgb_model,param_distributions=param_distributions,n_iter=10,scorint='accuracy',cv=3,verbose=1,n_jobs=-1,random_state=42)
rand_search.fit(X,y)

print("\n--- GridSearchCV Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

print("\n--- RandomizedSearchCV Results ---")
print(f"Best Parameters: {rand_search.best_params_}")
print(f"Best Score: {rand_search.best_score_:.4f}")