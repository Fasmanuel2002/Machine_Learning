import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el CSV
file_path = "iris2.csv"  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(file_path)

# Ver las primeras filas para identificar las columnas
print(df.head())

# Graficar todas las combinaciones de características coloreadas por la especie
sns.pairplot(df, hue=df.columns[-1], diag_kind="kde", palette="Set2")

# Mostrar el gráfico
plt.show()
