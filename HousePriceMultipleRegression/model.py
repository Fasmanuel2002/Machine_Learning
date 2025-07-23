from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
import numpy as np
import csv
from keras import Sequential
from keras import layers
import pandas as pd
from tensorflow.keras.optimizers import Adam # type: ignore #
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
TEST_SIZE = 0.30

# Predicciones
def main():
    df = pd.read_csv("augmented_cleaned_train.csv")
    
    X = df.iloc[:, 0:38].values
    Y =df["SalePrice"].values
    
    Y_log = np.log1p(Y) # type: ignore
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    
    #Normalizing
    
    
    x_train, x_test, y_train, y_test = train_test_split(
        X_norm, Y_log, test_size=TEST_SIZE, random_state=42
    )
    
    
    
    model = train_model(x_train.shape[1])
    model.fit(x_train, y_train, batch_size=64, epochs=200, validation_split=0.1)
    
    y_pred = model.predict(x_test)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(test_loss)
    print(test_accuracy)
    
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred.flatten())
    
    y_pred_train = model.predict(x_train).flatten()
    y_pred_test  = model.predict(x_test).flatten()


    y_train_orig = np.expm1(y_train)
    y_test_orig  = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig  = np.expm1(y_pred_test)

    # R²
    r2_train = r2_score(y_train_orig, y_pred_train_orig)
    r2_test  = r2_score(y_test_orig,  y_pred_test_orig)

    print(f"R² entrenamiento: {r2_train:.4f}")
    print(f"R² prueba:       {r2_test:.4f}")


    r2_orig = r2_score(y_test_orig, y_pred_orig)
    print(f"R² (escala real): {r2_orig:.4f}")
    if r2_orig >= 0.90:
        print("El modelo tiene la varianza esperada")
        model.save("HouseModel.h5")
    
    with open("scalerNormalizeHouse.pkl", "wb") as f1:
        pickle.dump(scaler, f1)
        
def train_model(input_shape) -> keras.Model:
    model = Sequential()
    
    model.add(layers.Input(shape=(input_shape, )))
    
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.BatchNormalization())
    layers.Dropout(0.8)
    model.add(layers.Dense(units=124, activation='relu'))
    model.add(layers.BatchNormalization())
    layers.Dropout(0.7)
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.BatchNormalization())
    layers.Dropout(0.4)
    model.add(layers.Dense(units=1, activation='linear'))
    
    
    #Output
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0015), # type: ignore
        loss='mse',
        metrics=['mae']
    )
    
    return model
    
if __name__ == "__main__":
    main()