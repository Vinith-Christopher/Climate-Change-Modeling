# Climate-Change-Modeling

Aim of this climate change Modeling is to make Machine Learning/ Deep Learning model to predict various aspect of climate change. From this we can predict future projections to make planing and mitigation efforts.

The Data used for is climate modelling is accessed from https://www.kaggle.com/code/sanau002/visualising-and-prediction-the-weather/input , from there we 
can access global temperature datas from global location by continent, country, city, state.


Preprocessing:
1. converting latitude and longitude columns to numerics, 
2. filling the missing values by Imputation method
3. encoding categorical column values to numeric columns values by one hot encoding 
4. evaluate temperature uncertainty ratio (from avg temperature uncertainty  and avg temperature)


Prediction:
preprocessed data is fed to the Machine learning classifiers and Deep Learning Models in order 
to perform Cross validation analysis  to evaluate performance of models.
