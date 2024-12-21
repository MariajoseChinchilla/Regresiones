# This class is made in order to fit linear regression models. Includes the following:
# Evaluation of correlation and combination of columns in plots
# One hot encoding for categorical columns, ols model, summary, anova table, 
# plot of residuals and predicted values
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from math import ceil

class RegressionPipeline:
    def __init__(self, df, categorical, formula, target):
        self.df = df
        self.categorical = categorical
        self.formula = formula
        self.target = target
        self.encoder = OneHotEncoder(drop="first", sparse=False)
        self.model = None
        self.encoded_columns = []
        self.df_numerical_cols = self.df.select_dtypes(include=["int64", "float64"])


    # Exploratory Data Analysis
    def eda(self):
        print("Description of data: ")
        print(self.describe())

        print("Information of data:")
        print(self.df.info())

        print("Correlation in columns:")
        print(self.df_numerical_cols.corr())

    # Returns dsubplots of combinations of all numerical columns with target
    def combined_scatter(self):
        fig, axes = plt.subplots(3, ceil(len(self.df_numerical_cols) / 3), figsize=(8,6))
        axes = axes.flatten()
        combinations = [(col, self.target) for col in self.df_numerical_cols.columns if col != self.target]
        
        for idx, (col1,col2) in enumerate(combinations):
            axes[idx].scatter(self.df_numerical_cols[col1], self.df_numerical_cols[col2], alpha=0.6)
            axes[idx].set_xlabel(col1)
            axes[idx].set_ylabel(col2)
            axes[idx].set_title(f"{col1} vs {col2}")
        
        plt.tight_layout()
        plt.show()

    
    def fit(self):
        # Step 1: encoding
        encoded = self.encoder.fit_transform(self.df[self.categorical])
        self.encoded_columns = self.encoder.get_feature_names_out(self.categorical)
        encoded_df = pd.DataFrame(encoded, columns=self.encoded_columns, index=self.df.index)

        # combine original df with encoded
        df_combined = pd.concat([self.df, encoded_df], axis=1)

        # adjust model with ols from stats formula
        self.model = ols(self.formula, data=df_combined).fit()
        self.df_combined = df_combined

    def summary(self, df):
        if self.model:
            print(self.model.summary())
        else:
            print("Model hasn't been fitted.")

    def anova(self):
        if self.model:
            anova_results = anova_lm(self.model)
            print(f"ANOVA: ")
            print(anova_results)
        else:
            print("Model hasn't been fitted yet.")

    def plot_diagnostic(self):
        if self.model:
            # residuals vs predicted
            residuals = self.model.resid
            predicted = self.model.ftted_values

            plt.figure(figsize=(8,6))
            plt.scatter(predicted, residuals, alpha=0.6)
            plt.xlabel("Valores predichos")
            plt.ylabel("Residuos")
            plt.title("Gráfico de residuos y valores predichos")
            plt.show()

    