import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Dataset de aumento de CO2 a nivel global 
data = {
    'Año': [1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 
            1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 
            1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 
            1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 
            1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 
            2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 
            2019, 2020, 2021, 2022],
    'CO2': [15.9, 15.9, 15.5, 15.6, 15.9, 16.2, 16.3, 16.6, 17.1, 17.8,
            18.3, 19.4, 20, 20.7, 21.4, 21.2, 21.3, 22.2, 22.8, 23.1,
            23.5, 23.4, 23.3, 23.1, 23.6, 24.8, 25.2, 25.6, 26.2, 27,
            27.1, 27.4, 27.8, 27.2, 27.4, 28, 28.5, 29.2, 31.1, 29.4,
            29.9, 30.3, 30.1, 31.1, 32.9, 33.6, 33.9, 35.2, 35.4, 36.2,
            36.1, 37.6, 38.8, 39.2, 39.3, 39.8, 40.2, 39.3, 39.7, 40.2,
            40.9, 38.5, 40.2, 40.5]
}

df = pd.DataFrame(data)

# Separar las características (X) y la variable objetivo (y)
X = df[['Año']]
y = df['CO2']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Aplicación Streamlit
st.title('Predicción de CO2 usando un Árbol de Decisión')

# Ajuste para permitir años futuros
year = st.number_input('Ingresa el año para predecir el nivel de CO2', min_value=1959, max_value=2050, step=1)

# Realizar la predicción
if st.button('Predecir'):
    prediction = model.predict([[year]])
    st.write(f'El nivel de CO2 estimado para el año {year} es: {prediction[0]:.2f}')
