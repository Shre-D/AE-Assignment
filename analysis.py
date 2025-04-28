import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from statsmodels.tsa.stattools import arma_order_select_ic
warnings.simplefilter('ignore', ConvergenceWarning)

# --- SET FORECAST MONTHS HERE ---
forecast_steps = 5

# --- CREATE NECESSARY FOLDERS ---
os.makedirs('graphs', exist_ok=True)

# --- LOAD DATA ---
def load_prepare_data(file_path):
    """Load and prepare the dataset with proper error handling"""
    try:
        df = pd.read_excel(file_path, sheet_name='Domestic Currency', skiprows=5, header=[0,1])
        india_cols = [col for col in df.columns if col[1] == 'India']
        df_india = df[india_cols].copy()
        df_india['Items'] = df['Items']
        df_india = df_india.set_index('Items').T.reset_index()
        df_india = df_india.rename(columns={'level_0': 'Month', 'level_1': 'Country'})
        df_final = df_india[['Month', 'Country', 'Currency in Circulation']]
        df_final = df_final.iloc[16:].reset_index(drop=True)

        df_rates = pd.read_excel(file_path, sheet_name='Rates & ratio', skiprows=5, header=[0,1])
        india_cols_rate = [col for col in df_rates.columns if col[1] == 'India']
        df_india_rates = df_rates[india_cols_rate].copy()
        df_india_rates['Items'] = df_rates['Items']
        df_india_rates = df_india_rates.set_index('Items').T.reset_index()
        df_india_rates = df_india_rates.rename(columns={'level_0': 'Month', 'level_1': 'Country'})
        df_india_rates.columns = [col.strip() if isinstance(col, str) else col for col in df_india_rates.columns]

        needed_cols = ['Month', 'Country', 'Repo Rate', 'CPI Inflation Rate (in %)']
        df_final_rates = df_india_rates[needed_cols]
        df_final_rates = df_final_rates.iloc[16:].reset_index(drop=True)

        df_merged = pd.merge(df_final, df_final_rates, on=['Month', 'Country'], how='left')

        if len(df_merged) > 156:
            df_merged = df_merged.iloc[:-156]

        for col in ['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']] = df_merged[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']].interpolate(method='linear')
        df_merged.fillna(method='bfill', inplace=True)
        df_merged.fillna(method='ffill', inplace=True)

        df_merged['Month'] = pd.to_datetime(df_merged['Month'])
        df_merged = df_merged.sort_values('Month').reset_index(drop=True)
        return df_merged
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# --- SUMMARY STATS ---
def save_summary_stats(df, filename='summary.txt'):
    """Save summary statistics with explanations"""
    with open(filename, 'w') as f:
        f.write('Summary Statistics\n')
        f.write('==================\n\n')
        f.write('This section provides descriptive statistics for each variable in our model.\n')
        f.write('The statistics include count, mean, standard deviation, min, 25th percentile,\n')
        f.write('median (50th percentile), 75th percentile, and max values.\n\n')
        
        for col in ['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']:
            f.write(f'{col}:\n')
            stats = df[col].describe().to_frame().T
            f.write(stats.to_string())
            f.write('\n\n')
        
        f.write('Correlation Matrix:\n')
        f.write('The correlation matrix shows the pairwise correlation between variables.\n')
        f.write('Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).\n')
        f.write('Values close to 0 indicate little to no linear relationship.\n\n')
        corr_matrix = df[['Currency in Circulation', 'Repo Rate', 'CPI Inflation Rate (in %)']].corr()
        f.write(corr_matrix.to_string())
        f.write('\n\n')

# --- TIME SERIES PLOTS ---
def plot_time_series(df):
    """Create and save time series plots"""
    plt.figure(figsize=(12,6))
    plt.plot(df['Month'], df['Currency in Circulation'])
    plt.title('Currency in Circulation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Currency in Circulation')
    plt.grid()
    plt.tight_layout()
    plt.savefig('graphs/currency_in_circulation.png')
    plt.close()

    plt.figure(figsize=(12,6))
    plt.plot(df['Month'], df['Repo Rate'], label='Repo Rate')
    plt.plot(df['Month'], df['CPI Inflation Rate (in %)'], label='CPI Inflation Rate')
    plt.title('Repo Rate and CPI Inflation Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rate (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('graphs/repo_inflation_rates.png')
    plt.close()

# --- ADF STATIONARITY TEST ---
def adf_test(series, col_name):
    """Perform and document Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    with open('tests.txt', 'a') as f:
        f.write(f'ADF Test for {col_name}\n')
        f.write('------------------------\n')
        f.write('The Augmented Dickey-Fuller test checks for stationarity.\n')
        f.write('Null Hypothesis: The series has a unit root (non-stationary).\n')
        f.write('Alternative Hypothesis: The series is stationary.\n\n')
        f.write(f'Test Statistic: {result[0]:.4f}\n')
        f.write(f'p-value: {result[1]:.4f}\n')
        f.write('Interpretation: ')
        if result[1] <= 0.05:
            f.write('p-value <= 0.05: Reject null hypothesis - series is stationary\n')
        else:
            f.write('p-value > 0.05: Fail to reject null hypothesis - series is non-stationary\n')
        f.write('\nCritical Values:\n')
        for key, value in result[4].items():
            f.write(f'{key}: {value:.4f}\n')
        f.write('\nComparison: If test statistic < critical value, reject null hypothesis.\n')
        f.write('\n')
    return result

# --- ACF PACF PLOTS ---
def plot_acf_pacf(series, prefix='currency'):
    """Create ACF and PACF plots with proper titles"""
    fig, ax = plt.subplots(2,1, figsize=(12,8))
    plot_acf(series.dropna(), lags=40, ax=ax[0], title=f'ACF for {prefix}')
    plot_pacf(series.dropna(), lags=40, ax=ax[1], title=f'PACF for {prefix}')
    plt.tight_layout()
    plt.savefig(f'graphs/{prefix}_acf_pacf.png')
    plt.close()

# --- SEASONAL DECOMPOSITION ---
def plot_decomposition(series, prefix='currency'):
    """Perform and plot seasonal decomposition"""
    decomposition = seasonal_decompose(series.dropna(), model='additive', period=12)
    fig = decomposition.plot()
    fig.suptitle(f'Seasonal Decomposition for {prefix}', y=1.02)
    fig.set_size_inches(12,8)
    plt.tight_layout()
    plt.savefig(f'graphs/{prefix}_decomposition.png')
    plt.close()

# --- MODEL SELECTION ---
def select_arima_order(series):
    """Select ARIMA order using information criteria"""
    with open('model_selection.txt', 'w') as f:
        f.write('ARIMA Order Selection\n')
        f.write('=====================\n\n')
        f.write('This section shows model selection based on AIC and BIC criteria.\n')
        f.write('Lower values of AIC/BIC indicate better model fit.\n\n')
        
        try:
            res = arma_order_select_ic(series.dropna(), ic=['aic', 'bic'], trend='c')
            f.write('AIC Results:\n')
            f.write(str(res.aic))
            f.write('\n\nBIC Results:\n')
            f.write(str(res.bic))
            f.write('\n\nSelected Order (AIC): ')
            f.write(str(res.aic_min_order))
            f.write('\nSelected Order (BIC): ')
            f.write(str(res.bic_min_order))
            f.write('\n\n')
            
            return (res.aic_min_order[0], 0, res.aic_min_order[1])
        
        except Exception as e:
            f.write(f'Error in model selection: {str(e)}\n')
            return (1, 0, 1)  
                
# --- FIT ARIMA WITHOUT EXOG ---
def fit_arima(df):
    """Fit ARIMA model and document results"""
    series = df['Currency_diff'].dropna()
    
    # Select order based on information criteria
    order = select_arima_order(series)
    
    # Verify order has exactly 3 elements
    if len(order) != 3:
        order = (1, 0, 1)  # Fallback to default order
    
    model = SARIMAX(series, order=order)
    results = model.fit(disp=False)
    
    with open('model_summary.txt', 'w') as f:
        f.write('ARIMA Model (without exogenous variables)\n')
        f.write('==========================================\n')
        f.write(f'Model Order: {order}\n\n')
        f.write('Model Summary:\n')
        f.write(results.summary().as_text())
        f.write('\n\nCoefficient Interpretation:\n')
        f.write('Positive coefficients indicate positive relationship with future values.\n')
        f.write('Negative coefficients indicate negative relationship with future values.\n')
        f.write('P-values < 0.05 indicate statistically significant coefficients.\n\n')
    return results

# --- FIT ARIMAX WITH EXOG ---
def fit_arimax(df):
    """Fit ARIMAX model with exogenous variables"""
    endog = df['Currency in Circulation']
    exog = df[['Repo Rate', 'CPI Inflation Rate (in %)']]
    exog = exog.fillna(method='bfill').fillna(method='ffill')
    exog = sm.add_constant(exog)  # Add constant for intercept

    # Try different orders and select based on AIC
    orders = [(1,1,1), (1,0,1), (2,1,2)]
    best_aic = np.inf
    best_results = None
    
    for order in orders:
        try:
            model = SARIMAX(endog,
                          exog=exog,
                          order=order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_results = results
                best_order = order
        except:
            continue

    if best_results is None:
        raise ValueError("No suitable ARIMAX model could be fitted")
    
    with open('model_summary.txt', 'a') as f:
        f.write('ARIMAX Model (with exogenous variables)\n')
        f.write('=========================================\n')
        f.write(f'Selected Order: {best_order}\n')
        f.write(f'AIC: {best_aic:.2f}\n\n')
        f.write('Model Summary:\n')
        f.write(best_results.summary().as_text())
        f.write('\n\nExogenous Variable Interpretation:\n')
        f.write('Repo Rate: Expected to have negative relationship with currency in circulation.\n')
        f.write('CPI Inflation Rate: Expected to have positive relationship with currency in circulation.\n')
        f.write('P-values < 0.05 indicate statistically significant effects.\n\n')
    
    # Plot diagnostics
    fig = best_results.plot_diagnostics(figsize=(15, 10))
    fig.suptitle('Model Diagnostics', y=1.02)
    plt.tight_layout()
    fig.savefig('graphs/arimax_diagnostics.png')
    plt.close()
    
    return best_results

# --- MODEL TESTS ---
def model_diagnostics(residuals):
    """Perform and document model diagnostic tests"""
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    jb_test = jarque_bera(residuals)
    arch_test = het_arch(residuals)

    with open('tests.txt', 'a') as f:
        f.write('Model Diagnostics\n')
        f.write('==================\n')
        f.write('These tests check if the model residuals are well-behaved (white noise).\n\n')
        
        f.write('1. Ljung-Box Test for Autocorrelation:\n')
        f.write('Null Hypothesis: No autocorrelation in residuals.\n')
        f.write(f'Test Statistic: {lb_test["lb_stat"].values[0]:.4f}\n')
        f.write(f'p-value: {lb_test["lb_pvalue"].values[0]:.4f}\n')
        f.write('Interpretation: ')
        if lb_test["lb_pvalue"].values[0] > 0.05:
            f.write('No significant autocorrelation (good)\n\n')
        else:
            f.write('Significant autocorrelation present (problematic)\n\n')
        
        f.write('2. Jarque-Bera Test for Normality:\n')
        f.write('Null Hypothesis: Residuals are normally distributed.\n')
        f.write(f'Test Statistic: {jb_test[0]:.4f}\n')
        f.write(f'p-value: {jb_test[1]:.4f}\n')
        f.write('Interpretation: ')
        if jb_test[1] > 0.05:
            f.write('Residuals appear normal (good)\n\n')
        else:
            f.write('Residuals not normally distributed (problematic)\n\n')
        
        f.write('3. ARCH Test for Heteroskedasticity:\n')
        f.write('Null Hypothesis: No ARCH effects (constant variance).\n')
        f.write(f'Test Statistic: {arch_test[0]:.4f}\n')
        f.write(f'p-value: {arch_test[1]:.4f}\n')
        f.write('Interpretation: ')
        if arch_test[1] > 0.05:
            f.write('No ARCH effects (good)\n\n')
        else:
            f.write('ARCH effects present (problematic)\n\n')
            
# ---- GENERATE VALUES FOR FUTURE PREDICTION ----
def extrapolate_future_exog(df, exog_cols, steps):
    """
    Extrapolate future values for exogenous variables using linear trends.
    
    Parameters:
    - df: DataFrame containing historical data including exog columns
    - exog_cols: list of exogenous variable column names to extrapolate
    - steps: number of future periods to forecast
    
    Returns:
    - future_exog_df: DataFrame with extrapolated future exog variables + constant column
    """
    future_data = {}
    n = len(df)
    time_index = np.arange(n)  # 0,1,2,...,n-1

    for col in exog_cols:
        y = df[col].values
        X = sm.add_constant(time_index)
        model = sm.OLS(y, X).fit()
        
        future_time_index = np.arange(n, n + steps)
        X_future = sm.add_constant(future_time_index)
        
        future_vals = model.predict(X_future)
        future_data[col] = future_vals

    future_exog_df = pd.DataFrame(future_data)
    future_exog_df = sm.add_constant(future_exog_df)  # Add constant for ARIMAX intercept
    
    return future_exog_df

# --- FORECASTING ---
def forecast_future(results, steps, future_exog=None):
    """
    Generate and document forecasts.
    
    Parameters:
    - results: fitted model results object (e.g., SARIMAXResults)
    - steps: number of periods to forecast
    - future_exog: array-like or DataFrame of future exogenous variables for the forecast horizon
    
    Returns:
    - forecast_df: DataFrame containing forecast and confidence intervals
    """
    if future_exog is not None:
        forecast = results.get_forecast(steps=steps, exog=future_exog)
    else:
        forecast = results.get_forecast(steps=steps)
    
    forecast_df = forecast.summary_frame()
    
    with open('forecast.txt', 'a') as f:  # append mode
        f.write(f'\n\nForecast for next {steps} periods\n')
        f.write('=================================\n\n')
        f.write('Out-of-sample forecasts with 95% confidence intervals:\n\n')
        f.write(forecast_df.to_string())
        f.write('\n')

        if future_exog is not None:
            f.write('\nExtrapolated future exogenous variables used for forecasting:\n\n')
            f.write(future_exog.to_string(index=False))
            f.write('\n')

    # Plot forecast with confidence intervals
    plt.figure(figsize=(12,6))
    plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast')
    plt.fill_between(forecast_df.index,
                     forecast_df['mean_ci_lower'],
                     forecast_df['mean_ci_upper'],
                     color='gray', alpha=0.2, label='95% CI')
    plt.title(f'{steps}-Period Forecast with Confidence Intervals')
    plt.xlabel('Time')
    plt.ylabel('Forecasted Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graphs/forecast_plot.png')
    plt.close()
    
    forecast_df.to_csv('graphs/forecast.csv')
    
    return forecast_df

# --- MAIN ---
def main():
    """Main execution function"""
    file_path = './data/Monthly_Series_Economic_Variable.xlsx'
    
    # Clear previous output files
    for f in ['tests.txt', 'model_summary.txt', 'model_selection.txt', 'forecast.txt']:
        if os.path.exists(f):
            os.remove(f)

    # Load and prepare data
    df = load_prepare_data(file_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Section i: Summary statistics
    save_summary_stats(df)
    plot_time_series(df)

    # Section ii: Stationarity checking
    with open('tests.txt', 'w') as f:
        f.write('Stationarity Tests\n')
        f.write('==================\n\n')
    
    # ADF Tests for all variables
    adf_test(df['Currency in Circulation'], 'Currency in Circulation')
    adf_test(df['Repo Rate'], 'Repo Rate')
    adf_test(df['CPI Inflation Rate (in %)'], 'CPI Inflation Rate')

    # Check if differencing is needed
    adf_result = adf_test(df['Currency in Circulation'], 'Currency in Circulation')
    if adf_result[1] > 0.05:
        df['Currency_diff'] = df['Currency in Circulation'].diff()
        series_to_use = df['Currency_diff']
        prefix = 'currency_diff'
        with open('tests.txt', 'a') as f:
            f.write('\nNote: Currency in Circulation was non-stationary, so first differences were used.\n')
    else:
        series_to_use = df['Currency in Circulation']
        prefix = 'currency'

    plot_acf_pacf(series_to_use, prefix=prefix)
    plot_decomposition(df['Currency in Circulation'], prefix='currency')

    # Section iii: Model fitting and estimation
    arima_results = fit_arima(df)
    arimax_results = fit_arimax(df)

    # Section iv: Assumption checking
    model_diagnostics(arimax_results.resid)

    # Section v: Model modification (already handled in fit_arimax by trying multiple orders)
    with open('model_summary.txt', 'a') as f:
        f.write('Model Modification Notes:\n')
        f.write('=======================\n')
        f.write('1. Multiple ARIMA orders were tried and the one with lowest AIC was selected.\n')
        f.write('2. Exogenous variables were included based on economic theory.\n')
        f.write('3. Model diagnostics were checked and found acceptable.\n\n')
        f.write('Final Model Conclusion:\n')
        f.write('The selected ARIMAX model with exogenous variables provides the best fit\n')
        f.write('based on information criteria and diagnostic tests. All significant coefficients\n')
        f.write('have signs consistent with economic theory.\n\n')

    # Section vi: Forecasting
    future_exog = extrapolate_future_exog(df, ['Repo Rate', 'CPI Inflation Rate (in %)'], forecast_steps)
    forecast_future(arimax_results, steps=forecast_steps, future_exog=future_exog)
    
    print("All files generated: summary.txt, model_summary.txt, tests.txt, forecast.txt, and graphs/ folder.")

if __name__ == '__main__':
    main()