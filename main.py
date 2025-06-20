import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.frequencies import to_offset
import openpyxl


st.set_page_config(page_title="Time Series Forecasting Suite", layout="wide")

# Define app sections
app_pages = ["About", "Data Upload & Analysis", "Forecasting"]
page = st.sidebar.radio("Navigation", app_pages)

# Global variables to store data across pages
if 'data' not in st.session_state:
    st.session_state.data = None

# ---------- PAGE 1: About ---------- #
if page == "About":
    st.title("📈 Time Series Forecasting Suite")
    st.image("https://martech.org/wp-content/uploads/2015/11/graph-line-trend-analytics-magnifying-glass-ss-1920.jpg", use_container_width=True)
    st.markdown("""
    Welcome to the **Time Series Forecasting Suite**, a powerful tool to help you:

    - Upload and analyze time series data from Excel or Google Sheets
    - Visualize trends of multiple variables
    - Apply advanced models like **ARIMA**, **SARIMA**, and **ETS** for forecasting
    - Download your forecast results

    ### Supported Forecasting Models
    - ARIMA: Auto-Regressive Integrated Moving Average
    - SARIMA: Seasonal ARIMA
    - ETS: Exponential Smoothing (Trend + Seasonality)

    This app is ideal for business analysts, data scientists, and researchers working on time-driven datasets.
    """)

# ---------- PAGE 2: Data Upload & Summary ---------- #
elif page == "Data Upload & Analysis":
    st.title("📊 Data Upload & Summary :")

    upload_method = st.radio("Select Upload Method", ["Upload Excel File", "Paste Google Sheet Link"])

    if upload_method == "Upload Excel File":
        uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.session_state.data = df

    elif upload_method == "Paste Google Sheet Link":
        gsheet_url = st.text_input("Paste Google Sheet URL")
        if gsheet_url:
            try:
                sheet_id = gsheet_url.split('/d/')[1].split('/')[0]
                gsheet_csv = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0'
                df = pd.read_csv(gsheet_csv)
                st.session_state.data = df
            except:
                st.error("Invalid or inaccessible Google Sheet link.")

    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0],errors='coerce')
        # df.iloc[:, 0] = df.iloc[:, 0].dt.strftime('%Y-%m')
        df = df.sort_values(by=df.columns[0])
        st.write("### Data Preview")
        st.dataframe(df)

        st.write("### Summary Statistics")
        st.write(df.describe())

        # var_to_plot = st.selectbox("Select variable to visualize", df.columns[1:])
        var_to_plot = st.selectbox("Select variable to visualize", df.columns)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df, x=df.columns[0], y=var_to_plot, ax=ax)
        ax.set_title(f"Trend of {var_to_plot}")
        st.pyplot(fig)

# ---------- PAGE 3: Forecasting ---------- #
elif page == "Forecasting":
    st.title("🔮 Forecasting")

    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values(by=df.columns[0])
        df.set_index(df.columns[0], inplace=True)

        model_choice = st.selectbox("Select Forecast Model", ["ARIMA", "SARIMA", "ETS"])
        target_variable = st.selectbox("Select variable to forecast", df.columns)
        forecast_months = st.number_input("Forecast Period (Months)", min_value=1, max_value=60, value=12)

        if model_choice in ["ARIMA", "SARIMA"]:
            exog_vars = st.multiselect("Select exogenous variables (optional)", [col for col in df.columns if col != target_variable])
            p = st.number_input("AR or seasonal order p", 0, 5, 1)
            d = st.number_input("Differencing order d", 0, 2, 1)
            q = st.number_input("MA order q", 0, 5, 1)
            if model_choice == "SARIMA":
                P = st.number_input("Seasonal AR P", 0, 5, 1)
                D = st.number_input("Seasonal diff D", 0, 2, 1)
                Q = st.number_input("Seasonal MA Q", 0, 5, 1)
                m = st.number_input("Seasonality (e.g., 12 for monthly)", 1, 24, 12)

        if st.button("Run Forecast"):
            end_date = df.index.max() + to_offset(f"{forecast_months}M")
            forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_months, freq='MS')

            if model_choice == "ARIMA":
                model = ARIMA(df[target_variable], order=(p, d, q), exog=df[exog_vars] if exog_vars else None)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_months, exog=np.zeros((forecast_months, len(exog_vars))) if exog_vars else None)

            elif model_choice == "SARIMA":
                model = SARIMAX(df[target_variable], order=(p, d, q), seasonal_order=(P, D, Q, m), exog=df[exog_vars] if exog_vars else None)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=forecast_months, exog=np.zeros((forecast_months, len(exog_vars))) if exog_vars else None)

            elif model_choice == "ETS":
                model = ExponentialSmoothing(df[target_variable], trend='add', seasonal='add', seasonal_periods=12)
                model_fit = model.fit()
                forecast = model_fit.forecast(forecast_months)

            forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast": forecast}).reset_index(drop=True)

            st.write("### Forecast Results")
            st.dataframe(forecast_df)

            st.write("### Forecast Trend")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=forecast_df, x="Date", y="Forecast", ax=ax)
            ax.set_title(f"Trend of {target_variable} Forecast")
            st.pyplot(fig)

            # df.to_excel(forecast_df, index=False)



            def to_excel_bytes(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()

            st.download_button(
                label="📥 Download Forecast as Excel",
                data=to_excel_bytes(forecast_df),
                file_name="forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Please upload data from the 'Data Upload & Analysis' section.")
