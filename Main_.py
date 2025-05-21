# --- Import Necessary python Modules

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Models_ import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit


def convert_lat_lon(val):
    """
    Convert latitude/longitude strings like '18.48N' or '82.82E' to float.
    'N' and 'E' are positive, 'S' and 'W' are negative.
    """
    if pd.isna(val):
        return np.nan
    direction = val[-1]
    number = float(val[:-1])
    if direction in ['S', 'W']:
        return -number
    return number



def Preprocess(visualize=False):
    # Load dataset
    df = pd.read_csv('DATASET/GlobalLandTemperaturesByCity.csv')

    # Filter for India and Visakhapatnam
    df = df[(df['Country'] == 'India') & (df['City'] == 'Visakhapatnam')]

    # Convert date to datetime
    df['dt'] = pd.to_datetime(df['dt'])

    # Extract year and month
    df['Year'] = df['dt'].dt.year
    df['Month'] = df['dt'].dt.month

    # Convert Latitude and Longitude to float
    df['Latitude'] = df['Latitude'].apply(convert_lat_lon)
    df['Longitude'] = df['Longitude'].apply(convert_lat_lon)

    # Drop rows with missing temperature
    df = df.dropna(subset=['AverageTemperature'])

    # Impute uncertainty with mean
    imputer = SimpleImputer(strategy='mean')
    df['AverageTemperatureUncertainty'] = imputer.fit_transform(df[['AverageTemperatureUncertainty']])

    # One-hot encode Month
    # df = pd.get_dummies(df, columns=['Month'], prefix='M')
    df = pd.get_dummies(df, columns=['Month', 'City', 'Country'])
    for col in df.select_dtypes(include=['object', 'category']):
        dummies = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1).join(dummies)
    df['Temp_Uncertainty_Ratio'] = df['AverageTemperatureUncertainty'] / (df['AverageTemperature'] + 1e-5)

    # Standardize numerical columns
    scaler = StandardScaler()
    df[[ 'AverageTemperatureUncertainty', 'Latitude', 'Longitude', 'Year']] = scaler.fit_transform(
        df[[ 'AverageTemperatureUncertainty', 'Latitude', 'Longitude', 'Year']]
    )

    # --- Visualizations
    if visualize:
        # Temperature trend over time
        plt.figure(figsize=(12, 5))
        plt.plot(df['dt'], df['AverageTemperature'], color='blue')
        plt.title('Temperature Trend Over Time - Visakhapatnam')
        plt.xlabel('YEAR', fontweight='bold')
        plt.ylabel('Standardized Temperature(\u00B0C)', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'Temperature Over Time.png')
    
        # Temperature distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(df['AverageTemperature'], kde=True, bins=30, color='orange')
        plt.title('Temperature Distribution')
        plt.tight_layout()
        plt.xlabel('Average Temperature(\u00B0C)', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.show()
        plt.savefig(f'Temperature Distribution.png')
    
        # Correlation 
        columns_ = ['AverageTemperature', 'AverageTemperatureUncertainty', 'Year', 'Temp_Uncertainty_Ratio']
        corr = df[columns_].corr()
        plt.figure(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation')
        plt.tight_layout()
        plt.show()
        plt.savefig('Correlation.png')



    df = df.drop(['dt'], axis=1)
    feat = df.drop(['AverageTemperature'], axis=1).values
    lab = df['AverageTemperature'].values
    return feat, lab

def cross_validate_method(method_class, method_name, X, y, folds=5):
    kf = TimeSeriesSplit(n_splits=5)
    all_metrics = []
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        model_instance = method_class(xtrain, xtest, ytrain, ytest)
        method = getattr(model_instance, method_name)
        metrics = method()
        all_metrics.append(metrics)

    metric_ = np.mean(all_metrics, axis=0).tolist()
    return metric_



def Analysis():
    # preprocessing
    feat, lab = Preprocess()
    # cross fold validation with 5 folds
    C1 = cross_validate_method(METHODS_, 'Linear_Regression', feat, lab)
    C2 = cross_validate_method(METHODS_, 'Decision_tree_', feat, lab)
    C3 = cross_validate_method(METHODS_, 'Random_Forest', feat, lab)
    C4 = cross_validate_method(METHODS_, 'XGBoost_', feat, lab)
    C5 = cross_validate_method(METHODS_, 'Neural_Network', feat, lab)

    comp = [C1, C2, C3, C4, C5]
    perf_names = ["MAE", "MSE", "RMSE", "R2"]  # Metrics
    # file name creation
    file_names = [f'{name}.npy' for name in perf_names]
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            new.append(comp[i][j])
        np.save(file_names[j], np.array(new))  # it will save as .npy file format



if __name__ == "__main__":
    Analysis()
    plot_metrics()
