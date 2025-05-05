# Hapus kolom ID karena tidak relevan
df.drop("id", axis=1, inplace=True)

# Isi nilai BMI yang kosong dengan median
df["bmi"].fillna(df["bmi"].median(), inplace=True)

# One-hot encoding untuk data kategorikal
df = pd.get_dummies(df, drop_first=True)

# Normalisasi fitur numerik
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(df[['age', 'avg_glucose_level', 'bmi']])

df.head()
