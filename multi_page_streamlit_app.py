
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (unchanged from your script)
@st.cache_data
def load_data():
    return pd.read_csv('Liang_Cleaned.csv')

df = load_data()

# Title of the App
st.markdown("<h1 style='text-align: center; color: #3a86ff;'>Comprehensive Petrol Analysis</h1>", unsafe_allow_html=True)


# Anomaly Detection
def anomaly_detection():
    st.header("Anomaly Detection")
    # Calculate IQR for 'Oil Production Rate' and identify anomalies
    Q1_oil = df['Oil Production Rate'].quantile(0.25)
    Q3_oil = df['Oil Production Rate'].quantile(0.75)
    IQR_oil = Q3_oil - Q1_oil
    lower_bound_oil = Q1_oil - 1.5 * IQR_oil
    upper_bound_oil = Q3_oil + 1.5 * IQR_oil
    anomalies_oil = df[(df['Oil Production Rate'] < lower_bound_oil) | (df['Oil Production Rate'] > upper_bound_oil)]
    st.subheader("Anomalies in Oil Production Rate")
    st.dataframe(anomalies_oil[['Well Name', 'Date', 'Oil Production Rate']])

    # Calculate IQR for 'Water Production Rate' and identify anomalies
    Q1_water = df['Water Production Rate'].quantile(0.25)
    Q3_water = df['Water Production Rate'].quantile(0.75)
    IQR_water = Q3_water - Q1_water
    lower_bound_water = Q1_water - 1.5 * IQR_water
    upper_bound_water = Q3_water + 1.5 * IQR_water
    anomalies_water = df[(df['Water Production Rate'] < lower_bound_water) | (df['Water Production Rate'] > upper_bound_water)]
    st.subheader("Anomalies in Water Production Rate")
    st.dataframe(anomalies_water[['Well Name', 'Date', 'Water Production Rate']])



# Comparison Analysis
def comparison_analysis():
    st.header("Comparison Analysis")

    # Aggregate fluid volumes for injector and producer wells
    total_volumes_by_type = df.groupby('Is Injector Well').agg({
        'Oil Production Rate': 'sum',
        'Water Production Rate': 'sum',
        'Gas Production Rate': 'sum',
        'Water Injection Rate': 'sum',
        'Gas Injection Rate': 'sum'
    }).rename(index={0: 'Producer', 1: 'Injector'})
    st.subheader("Aggregated Fluid Volumes by Well Type")
    st.dataframe(total_volumes_by_type)



# Correlation Analysis
def correlation_analysis():
    st.header("Correlation Analysis")
    correlation_matrix = df.corr()

    # Plotting the heatmap in Streamlit
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)



# Spatial Analysis
def spatial_analysis():
    st.header("Spatial Analysis")

    # Plotting the spatial distribution of wells
    colors = df['Is Injector Well'].map({0: 'blue', 1: 'red'})
    labels = df['Is Injector Well'].map({0: 'Producer', 1: 'Injector'})
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    for label, color in zip(['Producer', 'Injector'], ['blue', 'red']):
        mask = labels == label
        ax1.scatter(df['Surface X'][mask], df['Surface Y'][mask], c=color, label=label, s=100, edgecolors='black')
    ax1.set_title('Spatial Distribution of Wells')
    ax1.set_xlabel('Surface X')
    ax1.set_ylabel('Surface Y')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plotting the spatial patterns in oil production
    from scipy.interpolate import griddata
    xi = np.linspace(df['Surface X'].min(), df['Surface X'].max(), 500)
    yi = np.linspace(df['Surface Y'].min(), df['Surface Y'].max(), 500)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((df['Surface X'], df['Surface Y']), df['Oil Production Rate'], (xi, yi), method='cubic')

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    contour = ax2.contourf(xi, yi, zi, 15, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Oil Production Rate')

    ax2.scatter(df['Surface X'], df['Surface Y'], c=df['Oil Production Rate'], s=100, edgecolors='black', cmap='viridis')
    ax2.set_title('Spatial Patterns in Oil Production')
    ax2.set_xlabel('Surface X')
    ax2.set_ylabel('Surface Y')
    ax2.grid(True)
    st.pyplot(fig2)



# Time Series Analysis
def time_series_analysis():
    st.header("Time Series Analysis")

    df['Date'] = pd.to_datetime(df['Date'])
    df_timeseries = df.groupby('Date').agg({'Oil Production Rate': 'sum', 'Water Production Rate': 'sum'})

    # Plotting the time series of production rates
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    df_timeseries['Oil Production Rate'].plot(ax=ax3, label='Oil Production Rate')
    df_timeseries['Water Production Rate'].plot(ax=ax3, label='Water Production Rate')
    ax3.set_title('Time Series Analysis of Production Rates')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Production Rate')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)


# Trend Identification
def trend_identification():
    st.header("Trend Identification")

    df_trend = df.groupby('Date').agg({'Oil Production Rate': 'sum'})
    rolling_mean = df_trend['Oil Production Rate'].rolling(window=12).mean()
    rolling_std = df_trend['Oil Production Rate'].rolling(window=12).std()

    # Plotting the trend along with rolling mean and standard deviation
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    df_trend['Oil Production Rate'].plot(ax=ax4, label='Original')
    rolling_mean.plot(ax=ax4, label='Rolling Mean', color='red')
    rolling_std.plot(ax=ax4, label='Rolling Std', color='green')
    ax4.set_title('Rolling Mean & Rolling Standard Deviation')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Oil Production Rate')
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4)


# Well Performance Analysis
def well_performance_analysis():
    st.header("Well Performance Analysis")

    top_producers = df.groupby('Well Name').agg({'Oil Production Rate': 'sum'}).nlargest(5, 'Oil Production Rate')

    # Plotting the top 5 oil producers
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    top_producers.plot(kind='bar', legend=False, ax=ax5)
    ax5.set_title('Top 5 Oil Producers')
    ax5.set_ylabel('Cumulative Oil Production Rate')
    ax5.set_xlabel('Well Name')
    plt.tight_layout()
    st.pyplot(fig5)


# Optimization Analysis
def optimization_analysis():
    st.header("Optimization Analysis")

    # Displaying the correlation between Water Injection Rate and Oil Production Rate
    correlation_water_oil = df[['Water Injection Rate', 'Oil Production Rate']].corr().iloc[0, 1]
    st.write(f"Correlation between Water Injection Rate and Oil Production Rate: {correlation_water_oil:.2f}")

    # Plotting the spatial distribution of wells by cumulative oil production
    cumulative_oil_production = df.groupby('Well Name')['Oil Production Rate'].sum().reset_index()
    well_location_oil = df[['Well Name', 'Surface X', 'Surface Y']].drop_duplicates().merge(cumulative_oil_production, on='Well Name')
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Surface X', y='Surface Y', hue='Oil Production Rate', size='Oil Production Rate',
                    sizes=(50, 500), data=well_location_oil, palette="YlOrRd", legend="full", ax=ax6)
    ax6.set_title('Spatial Distribution of Wells by Cumulative Oil Production')
    ax6.set_xlabel('Surface X Coordinate')
    ax6.set_ylabel('Surface Y Coordinate')
    ax6.legend(title='Cumulative Oil Production')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig6)



# Predictive Model
def predictive_model():
    st.header("Oil Production Forecasting")
    st.write("This page provides an analysis and forecasting tool for oil production rates based on historical data.")

    df = load_data()
    st.write("Here's a sample of the dataset:")
    st.write(df.sample(5))

    # Display basic statistics and visualizations of the dataset
    st.write("Basic statistics of the dataset:")
    st.write(df.describe())

    # Cumulative oil production for each well
    cumulative_oil_production = df.groupby("Well Name")["Oil Production Rate"].sum().sort_values(ascending=False)
    st.write("Top 5 wells by cumulative oil production:")
    st.bar_chart(cumulative_oil_production.head(5))

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.model_selection import train_test_split
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    # Allow users to select a well from the dataset
    selected_well = st.selectbox("Select a well to analyze:", df['Well Name'].unique())

    well_data = df[df["Well Name"] == selected_well]

    # View the production rate over time for the selected well
    st.write(f"Oil Production Rate Over Time for Well {selected_well}")
    st.line_chart(well_data.set_index('Date')['Oil Production Rate'])

    # Implement and visualize the ARIMA forecast for the selected well
    st.write("ARIMA Model Forecasting")

    # User interaction components for refining the ARIMA model parameters
    p = st.slider("Select ARIMA parameter p (recommended: 2):", 0, 7, 1)
    d = st.slider("Select ARIMA parameter d (recommended: 2):", 0, 2, 1)
    q = st.slider("Select ARIMA parameter q (recommended: 0):", 0, 7, 1)

    oil_production_rate = well_data["Oil Production Rate"]
    train, test = train_test_split(oil_production_rate, test_size=0.15, shuffle=False)

    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))
    forecast_series = pd.Series(forecast, index=test.index)

    st.write("Historical and Forecasted Oil Production Rate")
    fig7, ax = plt.subplots(figsize=(10, 6))
    train.plot(ax=ax, label="Historical Data")
    test.plot(ax=ax, label="Actual Test Data")
    forecast_series.plot(ax=ax, label="Forecasted Data", linestyle="--")
    ax.legend()
    st.pyplot(fig7)

# Sidebar with page selection
page_selection = st.sidebar.selectbox("Choose a page:", ["Anomaly Detection", "Comparison Analysis", "Correlation Analysis", "Spatial Analysis", "Time Series Analysis", "Trend Identification", "Well Performance Analysis", "Optimization Analysis", "Predictive Model"])

# Display the selected page
if page_selection == "Anomaly Detection":
    anomaly_detection()
elif page_selection == "Comparison Analysis":
    comparison_analysis()
elif page_selection == "Correlation Analysis":
    correlation_analysis()
elif page_selection == "Spatial Analysis":
    spatial_analysis()
elif page_selection == "Time Series Analysis":
    time_series_analysis()
elif page_selection == "Trend Identification":
    trend_identification()
elif page_selection == "Well Performance Analysis":
    well_performance_analysis()
elif page_selection == "Optimization Analysis":
    optimization_analysis()
elif page_selection == "Predictive Model":
    predictive_model()
