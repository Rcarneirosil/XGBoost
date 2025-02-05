import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gerando dados de série temporal fictícios
np.random.seed(42)
data = pd.DataFrame({
    'day': np.arange(1, 101),
    'value': np.sin(np.arange(1, 101) / 10) + np.random.normal(scale=0.2, size=100)
})

# Separação treino/teste
X = data[['day']]
y = data['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Sidebar para hiperparâmetros do XGBoost
st.sidebar.header("Ajuste os Hiperparâmetros")
n_estimators = st.sidebar.slider("Número de Árvores (n_estimators)", 10, 500, 100)
learning_rate = st.sidebar.slider("Taxa de Aprendizado (learning_rate)", 0.01, 0.5, 0.1)
max_depth = st.sidebar.slider("Profundidade Máxima (max_depth)", 2, 10, 3)

# Treinando o modelo XGBoost
model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Exibir gráfico e erro
st.subheader("Previsão de Séries Temporais com XGBoost")
st.line_chart(pd.DataFrame({'Real': y_test.values, 'Previsto': y_pred}, index=X_test.index))

st.write(f"Erro Quadrático Médio (MSE): {mse:.4f}")
