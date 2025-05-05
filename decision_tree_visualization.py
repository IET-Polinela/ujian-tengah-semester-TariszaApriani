import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Ukuran gambar yang lebih besar untuk pohon yang kompleks
plt.figure(figsize=(25, 15))

# Visualisasi pohon keputusan dengan penyesuaian
plot_tree(
    dt,  # Model Decision Tree yang telah dilatih
    feature_names=X.columns,  # Nama-nama fitur
    class_names=['No Stroke', 'Stroke'],  # Nama-nama kelas
    filled=True,  # Pewarnaan node berdasarkan kelas mayoritas
    rounded=True,  # Sudut node dibulatkan
    fontsize=10,  # Ukuran font
    proportion=True  # Menampilkan proporsi kelas dalam setiap node
)

# Menambahkan judul pada visualisasi
plt.title("Visualisasi Pohon Keputusan untuk Prediksi Risiko Stroke", fontsize=16)

# Menyimpan visualisasi ke file PNG dengan resolusi tinggi
plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches='tight')

# Menampilkan visualisasi di output
plt.show()
