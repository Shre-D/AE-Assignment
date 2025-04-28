import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.simplefilter('ignore', ConvergenceWarning)

# Create folder for graphs if it doesn't exist
os.makedirs('graphs', exist_ok=True)

def load_prepare_data(file_path):
    # Load Domestic Currency sheet with multi-index headers, skip first 5 rows
    df = pd.read_excel(file_path, sheet_name='Domestic Currency', skiprows=5, header=[0,1])
    
    # Select only columns where second level header is 'India'
    india_cols = [col for col in df.columns if col[1] == 'India']
    df_india = df[india_cols].copy()
    
    # Add 'Items' column to index to reshape the data
    df_india['Items'] = df['Items']
    
    # Transpose so that months become rows and Items become columns
    df_india = df_india.set_index('Items').T.reset_index()
    df_india = df_india.rename(columns={'level_0': 'Month', 'level_1': 'Country'})

    # Extract relevant columns for currency data and skip first 16 rows (usually header/metadata)
    df_final = df_india[['Month', 'Country', 'Currency in Circulation']]
    df_final = df_final.iloc[16:].reset_index(drop=True)

    # Load Rates & Ratio sheet similarly
    df_rates = pd.read_excel(file_path, sheet_name='Rates & ratio', skiprows=5, header=[0,1])
    india_cols_rate = [col for col in df_rates.columns if col[1] == 'India']
    df_india_rates = df_rates[india_cols_rate].copy()
    df_india_rates['Items'] = df_rates['Items']
    df_india_rates = df_india_rates.set_index('Items').T.reset_index()
    df_india_rates = df_india_rates.rename(columns={'level_0': 'Month', 'level_1': 'Country'})

    # Strip whitespace in column names to avoid key errors
    df_india_rates.columns = [col.strip() if isinstance(col, str) else col for col in df_india_rates.columns]

    # Verify required columns exist
    needed_cols = ['Month', 'Country', 'Repo Rate', 'CPI Inflation Rate (in %)']
    missing_cols = [c for c in needed_cols if c not in df_india_rates.columns]
    if missing_cols:
        raise KeyError(f"Expected columns not found: {missing_cols}")

    df_final_rates = df_india_rates[needed_cols]
    df_final_rates = df_final_rates.iloc[16:].reset_index(drop=True)

    # Merge currency and rate data on Month and Country
    df_merged = pd.merge(df_final, df_final_rates, on=['Month', 'Country'], how='left')

    # Drop last 156 rows if dataset is large, else comment out or parameterize
    if len(df_merged) > 156:
        df_merged = df_merged.iloc[:-156]

    # Convert columns to numeric, coercing non-numeric entries to NaN (e.g. '--')
    for col in ['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    # Replace infinite values with NaN as a safety measure
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Linearly interpolate missing values to maintain time series continuity
    df_merged[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']] = \
        df_merged[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']].interpolate(method='linear')

    # Forward and backward fill remaining NaNs (usually at edges)
    df_merged.fillna(method='bfill', inplace=True)
    df_merged.fillna(method='ffill', inplace=True)

    # Convert Month to datetime for time series operations
    df_merged['Month'] = pd.to_datetime(df_merged['Month'])
    df_merged = df_merged.sort_values('Month').reset_index(drop=True)

    # Warning if NaNs remain in key series after filling
    if df_merged[['Repo Rate', 'CPI Inflation Rate (in %)']].isnull().any().any():
        print("Warning: NaNs remain in Repo Rate or CPI Inflation Rate after filling!")

    return df_merged


def save_summary_stats(df, filename='summary.txt'):
    with open(filename, 'w') as f:
        f.write('Summary Statistics\n')
        f.write('==================\n\n')
        f.write('Currency in Circulation:\n')
        f.write(str(df['Currency in Circulation'].describe()))
        f.write('\n\nRepo Rate:\n')
        f.write(str(df['Repo Rate'].describe()))
        f.write('\n\nCPI Inflation Rate (in %):\n')
        f.write(str(df['CPI Inflation Rate (in %)'].describe()))
        f.write('\n\nCorrelation Matrix:\n')
        
        corr = df[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']].corr()
        f.write(str(corr))
        f.write('\n\n')


def plot_time_series(df):
    # Plot Currency in Circulation separately due to large scale (~1e7)
    plt.figure(figsize=(12,6))
    plt.plot(df['Month'], df['Currency in Circulation'], label='Currency in Circulation', color='tab:blue')
    plt.title('Currency in Circulation Over Time')
    plt.xlabel('Month')
    plt.ylabel('Currency in Circulation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graphs/currency_in_circulation.png')
    plt.close()

    # Plot Repo Rate and CPI Inflation Rate together, similar scale
    plt.figure(figsize=(12,6))
    plt.plot(df['Month'], df['Repo Rate'], label='Repo Rate', color='tab:orange')
    plt.plot(df['Month'], df['CPI Inflation Rate (in %)'], label='CPI Inflation Rate (%)', color='tab:green')
    plt.title('Repo Rate and CPI Inflation Rate Over Time')
    plt.xlabel('Month')
    plt.ylabel('Rate / Inflation (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graphs/repo_inflation_rates.png')
    plt.close()


def adf_test(series, col_name):
    result = adfuller(series.dropna())
    print(f'ADF Test for {col_name}:')
    print(f'  Test Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print('  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.4f}')
    print('---\n')
    return result


def plot_acf_pacf(series, lags=40, prefix='currency'):
    fig, ax = plt.subplots(2,1, figsize=(12,8))
    plot_acf(series.dropna(), lags=lags, ax=ax[0])
    plot_pacf(series.dropna(), lags=lags, ax=ax[1])
    ax[0].set_title(f'ACF Plot for {prefix}')
    ax[1].set_title(f'PACF Plot for {prefix}')
    plt.tight_layout()
    plt.savefig(f'graphs/{prefix}_acf_pacf.png')
    plt.close()


def plot_decomposition(series, period=12, prefix='currency'):
    decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12,8)
    plt.tight_layout()
    plt.savefig(f'graphs/{prefix}_decomposition.png')
    plt.close()


def fit_arimax_model(df):
    print("\nFitting ARIMAX model...\n")

    endog = df['Currency in Circulation']
    exog = df[['Repo Rate', 'CPI Inflation Rate (in %)']]

    # Fill NaNs in exogenous variables for the model
    exog = exog.fillna(method='bfill').fillna(method='ffill')

    model = SARIMAX(endog,
                    exog=exog,
                    order=(1,1,1),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)

    with open('model_summary.txt', 'w') as f:
        f.write(results.summary().as_text())

    print("Model fitted. Summary saved to model_summary.txt")

    fig = results.plot_diagnostics(figsize=(15, 10))
    plt.tight_layout()
    fig.savefig('graphs/arimax_diagnostics.png')
    plt.close()


def main():
    file_path = './Monthly_Series_Economic_Variable.xlsx'
    df = load_prepare_data(file_path)

    save_summary_stats(df)
    plot_time_series(df)

    adf_result = adf_test(df['Currency in Circulation'], 'Currency in Circulation')

    if adf_result[1] > 0.05:
        print("Series is non-stationary, differencing the series.")
        df['Currency_diff'] = df['Currency in Circulation'].diff()
        series_to_use = df['Currency_diff']
        prefix = 'currency_diff'
    else:
        print("Series is stationary.")
        series_to_use = df['Currency in Circulation']
        prefix = 'currency'

    plot_acf_pacf(series_to_use, prefix=prefix)
    plot_decomposition(df['Currency in Circulation'], prefix='currency')

    fit_arimax_model(df)

    print("All plots saved in 'graphs/' folder.")
    print("Summary statistics saved in 'summary.txt'.")
    print("Model summary saved in 'model_summary.txt'.")


if __name__ == '__main__':
    main()
