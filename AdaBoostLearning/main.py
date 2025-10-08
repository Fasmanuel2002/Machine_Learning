import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Generar un dataset sintético grande
X, y = make_classification(
    n_samples=10000,   # 10,000 muestras
    n_features=10,     # 10 características
    n_informative=7,   # 7 características útiles
    n_redundant=3,     # 3 características redundantes
    n_classes=2,       # clasificación binaria
    random_state=42
)

# Convertir etiquetas a -1 y 1 (como en tu ejemplo)
y = np.where(y == 0, -1, 1)

# Crear un DataFrame
df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
df['y'] = y

# 2. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], test_size=0.2, random_state=42)

# 3. Definir AdaBoost con stumps
stump = DecisionTreeClassifier(max_depth=1)  # Decision stump
ada = AdaBoostClassifier(stump, n_estimators=50, random_state=42)

# 4. Entrenar AdaBoost
ada.fit(X_train, y_train)

# 5. Predecir en el conjunto de prueba
y_pred = ada.predict(X_test)

# 6. Evaluar precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión en el conjunto de prueba: {accuracy:.4f}')
