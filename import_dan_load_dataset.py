# Import library yang dibutuhkan
import pandas as pd
import numpy as np

# Upload file dari lokal
from google.colab import files
uploaded = files.upload()

# Load data ke DataFrame
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Tampilkan 5 baris pertama
df.head()

# Info umum
df.info()

# Statistik deskriptif
df.describe()

# Cek missing value
df.isnull().sum()

