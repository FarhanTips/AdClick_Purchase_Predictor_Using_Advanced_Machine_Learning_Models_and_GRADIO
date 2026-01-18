import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import cross_val_score, GridSearchCV


df = pd.read_csv("Social_Network_Ads.csv")

if 'User ID' in df.columns:
  df.drop( columns = ['User ID'], inplace=True )

X = df.drop('Purchased',axis=1)
y = df['Purchased']

numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


Best_voting_cls = VotingClassifier(
    estimators=[
        ('log', LogisticRegression(max_iter=1000, random_state=42)),
        ('rfc', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)),
        ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42))
    ],
    voting='soft' 
)

Best_voting_pipe = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', Best_voting_cls)
])

Best_voting_pipe.fit(X_train, y_train)

y_pred = Best_voting_pipe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nResult:\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


filename = "Voting_ensemble_model_Best.pkl"

with open( filename, "wb" ) as file:
  pickle.dump( Best_voting_pipe, file )


print("\nPickle File Saved.")

