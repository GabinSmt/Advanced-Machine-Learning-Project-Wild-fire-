import pandas as pd

df=pd.read_csv("Wildfire_Dataset.csv")
df.head(76)

type(df.iloc[0]['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'])
type(df.iloc[0]['datetime'])

block_size = 75

# Liste pour stocker les résultats
rows = []

# Boucle sur les blocs de 75 lignes
for start in range(0, len(df), block_size):
    subset = df.iloc[start:start + block_size]
    mean_values = subset.iloc[:60].select_dtypes('number').mean()
    position = (subset['latitude'].iloc[0], subset['longitude'].iloc[0])
    date_label = subset['datetime'].iloc[0]
    fire_value = subset['Wildfire'].iloc[60]
    wildfire_tuple = (0, fire_value) 
    row = {
        'position': position,
        'datetime': date_label,
        'Wildfire': wildfire_tuple
    }
    for col, val in mean_values.items():
        row[col] = val
    rows.append(row)

df_aggregated = pd.DataFrame(rows)


print(df.shape)
print(df.shape[0]/75)
print(df_aggregated.shape)
df_aggregated.head()

df_aggregated.to_csv("df_aggregated.csv", index=False) 
