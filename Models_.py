# ---------- Import Necessary Python Modules -----------
import numpy as np
import pandas as pd
import xgboost as xgb
from keras import Model, Input, Sequential
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from termcolor import colored


def main_est_perf_metrics(y_true, pred):
    # Mean Absolute Error
    MAE = np.mean(abs(y_true - pred))
    # Mean Squared Error
    MSE = mean_squared_error(y_true, pred)
    # Root Mean Squared Error
    RMSE = np.sqrt(MSE)
    # coefficient of determination
    R2 = r2_score(y_true, pred)
    return [MAE, MSE, RMSE, R2]


class METHODS_:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def plot_last_10_predictions(self, ytest, pred):
        # Convert to numpy arrays in case they aren't
        ytest = np.array(ytest)
        pred = np.array(pred)
        last_10_actual = ytest[-10:]
        last_10_pred = pred[-10:]
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(10), last_10_actual, label='Actual', marker='o', ms=10, mec='k', linestyle='-')
        plt.plot(range(10), last_10_pred, label='Predicted', marker='x', ms=10, mec='k', linestyle='--')
        plt.title("Future Projection for 10 Days", fontsize=14, fontweight='bold')
        plt.xlabel("Days", fontsize=12,fontweight='bold')
        plt.ylabel("Average Temperature(\u00B0C)", fontsize=12, fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        plt.show()

    def Linear_Regression(self):
        lr = LinearRegression()
        lr.fit(self.xtrain, self.ytrain)
        pred = lr.predict(self.xtest)
        metrics = main_est_perf_metrics(self.ytest, pred)
        # Plot projection for last 10 test data
        self.plot_last_10_predictions(self.ytest, pred)
        return metrics

    def Decision_tree_(self):
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(self.xtrain, self.ytrain)
        pred = dt.predict(self.xtest)
        metrics = main_est_perf_metrics(self.ytest, pred)
        return metrics

    def Random_Forest(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.xtrain, self.ytrain)
        pred = rf.predict(self.xtest)
        metrics = main_est_perf_metrics(self.ytest, pred)
        return metrics

    def XGBoost_(self):
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(self.xtrain, self.ytrain)
        pred = xgb_model.predict(self.xtest)
        metrics = main_est_perf_metrics(self.ytest, pred)
        return metrics

    def Neural_Network(self):
        # Standardize the data
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(self.xtrain)
        xtest = scaler.transform(self.xtest)


        input_layer = Input(shape=(xtrain.shape[1],))
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=['MAE'])
        model.fit(xtrain, self.ytrain, epochs=50, batch_size=8)

        # Prediction
        pred = model.predict(xtest).flatten()
        metrics = main_est_perf_metrics(self.ytest, pred)

        self.plot_last_10_predictions(self.ytest, pred)
        return metrics


def plot_metrics(val, M):
    # val should be a 1D array-like input of metric values
    metrics = ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']

    filename_png = f'Visualizations/comp_{M}.png'
    x = np.arange(len(metrics))  # x-axis positions
    bar_width = 0.6
    colors = ['#e6ac00', '#666699', '#999966', '#00ffcc', '#0099ff', '#ac3973']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, val, width=bar_width, color=colors, edgecolor='black', linewidth=3,  alpha=1)


    for bar in bars:
        yval = bar.get_height()
        if M == 'Coefficient of Determination':
            max_value = max(val)
            color = 'green' if yval == max_value else 'red'
        else:
            min_value = min(val)
            color = 'green' if yval == min_value else 'red'
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}',
                 ha='center', va='bottom', fontsize=15, fontweight='bold', color=color)

    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel(M, fontsize=14, fontweight='bold')
    plt.xticks(x, metrics, rotation=30, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')
    plt.ylim(0, max(val) + 0.06)
    plt.tight_layout()
    plt.savefig(filename_png, dpi=800)
    plt.show()

def Plot_graphs(plot=False):
    if plot:
      mae_ = np.load('MAE.npy')
      mse_= np.load('MSE.npy')
      rmse_= np.load('RMSE.npy')
      R2_ = np.load('R2.npy')
  
      # ----------- saved to csv ----------------
      metrics = ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
      # Create DataFrame
      df = pd.DataFrame({
          'Model': metrics,
          'MAE': mae_,
          'MSE': mse_,
          'RMSE': rmse_,
          'R2': R2_
      })
      # Save to CSV
      df.to_csv('model_performance_metrics.csv', index=False)
  
      # ----------------- plot bar graphs ---------------
      plot_metrics(mae_, 'Mean Absolute Error')
      plot_metrics(mse_, 'Mean Squared Error')
      plot_metrics(rmse_, 'Root Mean Squared Error')
      plot_metrics(R2_, 'Coefficient of Determination')


