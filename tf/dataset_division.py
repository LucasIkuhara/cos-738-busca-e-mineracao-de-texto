# %%
import pandas as pd

# %%

# Divide 20 percent of the dataset for testing and 80% for training
df = pd.read_csv('raw_data.csv')
df = df.dropna()

div = int(len(df) * 0.8)

train = df.iloc[0:div]
test = df.iloc[div:]

print(f"Train: {len(train)}\nTest: {len(test)}")
train.to_csv("train_raw.csv", index=False)
test.to_csv("test_raw.csv", index=False)

# %%
# Compute stats
df2 = df.groupby('status').count()
df2['quatidade'] = df2['id']
df2['freq'] = df2['id'] / len(df)
df.drop(columns=['id', 'statement'])

df2.head(10)
