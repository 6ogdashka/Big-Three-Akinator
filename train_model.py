import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('demo-Blich-data.csv')
df_copy = df
X = df.drop('Character', axis = 1)
names = df['Character']

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
#model = RandomForestClassifier(n_estimators=170, random_state=42)
#model.fit(X,names)
#importance = pd.Series(model.feature_importances_, index=X.columns)
#importance.sort_values(ascending=False).to_csv('demo-feature_importance.csv', header=True)

#ask_que = []
user_responses = pd.Series(-1, index=X.columns)
#df_feature = pd.read_csv('demo-feature_importance.csv')

def transf():
    return np.random.choice([1.0,1.0,1.0,0.0,-0.7,-0.3,-1.0])

user_responses = user_responses.apply(lambda val:transf())

print(np.var(user_responses))



