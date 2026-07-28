import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("demo-Blich-data-augmented.csv")

X = df.drop("Character", axis=1).astype(float)
names = df["Character"]

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
model = RandomForestClassifier(n_estimators=170, random_state=42)
model.fit(X,names)
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).to_csv('demo-feature-importance.csv', header=True)
