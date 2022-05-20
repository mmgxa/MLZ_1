# MLZ Module 1

See [the notebook here!](./W1.ipynb)


# Q1

```
df_train = pd.read_parquet('../../dataset/fhv_tripdata_2021-01.parquet')
print(len(df_train))
```

Ans: 1154112

# Q2



```
df_train['duration'] = df_train.dropOff_datetime - df_train.pickup_datetime
df_train.duration = df_train.duration.apply(lambda td: td.total_seconds() / 60)
df_train.duration.mean()
```

Ans: 19.16

# Q3

```
frac = df_train.PUlocationID.isnull().sum() / len(df_train) * 100
frac
```
Answer: 83%


# Q4

```
categorical = ['PUlocationID', 'DOlocationID']
df_train[categorical] = df_train[categorical].astype(str)
dv = DictVectorizer()

train_dicts = df_train[categorical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

X_train.shape # either from here
len(dv.get_feature_names_out()) # or here
```
Answer: 525


# Q5

```
target = 'duration'
y_train = df_train[target].values
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
mean_squared_error(y_train, y_pred, squared=False)
```

Ans: 10.52

# Q6

```
df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)]

df_val['duration'] = df_val.dropOff_datetime - df_val.pickup_datetime
df_val.duration = df_val.duration.apply(lambda td: td.total_seconds() / 60)

categorical = ['PUlocationID', 'DOlocationID']
df_val[categorical] = df_val[categorical].astype(str)

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

target = 'duration'
y_val = df_val[target].values

y_pred_val = lr.predict(X_val)
mean_squared_error(y_val, y_pred_val, squared=False)
```
Answer: 11.01