import pandas as pd
import numpy as np

df = pd.read_csv("Blich-data.csv")
df = df.replace({0: -1, 1: 1})

def transform_value(val):
    if val == 1:
        return np.random.choice([1.0, 0.7, 0.3, 0.0])
    elif val == -1:
        return np.random.choice([-1.0, -0.7, -0.3, 0.0])
    return val

new_df = df.apply(lambda row: row.map(transform_value), axis=1)
second_df = df.apply(lambda row: row.map(transform_value), axis=1)

final_df = pd.concat([df, new_df, second_df], axis=0, ignore_index=True)
final_df.to_csv("demo-Blich-data.csv", index=False)

