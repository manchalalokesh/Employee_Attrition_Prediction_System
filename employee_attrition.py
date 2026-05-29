# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load(
    'employee_attrition_model.pkl'
)

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(

    "Select Menu",

    [
        "Dashboard",
        "Prediction",
        "Upload Dataset",
        "Analytics"
    ]
)

# -------------------------------
# TITLE
# -------------------------------

st.title("📊 Employee Attrition Prediction Dashboard")

st.markdown("---")

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "Dashboard":

    st.subheader("📈 HR Business Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        "1470"
    )

    col2.metric(
        "Attrition Rate",
        "16%"
    )

    col3.metric(
        "High Risk Employees",
        "237"
    )

    col4.metric(
        "Retention Score",
        "84%"
    )

    st.markdown("---")

    # Risk Distribution Data
    risk_data = pd.DataFrame({

        'Risk Level': [
            'Low Risk',
            'Medium Risk',
            'High Risk'
        ],

        'Employees': [
            900,
            330,
            240
        ]
    })

    # Pie Chart
    fig1 = px.pie(

        risk_data,

        values='Employees',

        names='Risk Level',

        title='Employee Risk Distribution'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # Department Attrition
    dept_data = pd.DataFrame({

        'Department': [
            'Sales',
            'HR',
            'IT',
            'Finance',
            'Marketing'
        ],

        'Attrition': [
            120,
            45,
            60,
            30,
            55
        ]
    })

    fig2 = px.bar(

        dept_data,

        x='Department',

        y='Attrition',

        title='Department Wise Attrition'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================================
# PREDICTION
# ==========================================================

elif menu == "Prediction":

    st.subheader("🧠 Employee Attrition Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            18,
            60
        )

        monthly_income = st.number_input(
            "Monthly Income"
        )

        overtime = st.selectbox(
            "OverTime",
            [0, 1]
        )

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4
        )

    with col2:

        years_company = st.number_input(
            "Years At Company"
        )

        work_life = st.slider(
            "Work Life Balance",
            1,
            4
        )

        environment = st.slider(
            "Environment Satisfaction",
            1,
            4
        )

        training = st.slider(
            "Training Times Last Year",
            0,
            10
        )

    if st.button("Predict Attrition"):

        # Input Array
        input_data = np.array([

            age,
            monthly_income,
            overtime,
            job_satisfaction,
            years_company,
            work_life,
            environment,
            training

        ]).reshape(1, -1)

        # Prediction
        prediction = model.predict(
            input_data
        )

        # Probability
        probability = model.predict_proba(
            input_data
        )[0][1]

        # Risk Classification
        if probability <= 0.30:

            risk = "Low Risk"

            recommendation = (
                "Career Growth Opportunities"
            )

        elif probability <= 0.60:

            risk = "Medium Risk"

            recommendation = (
                "Monitor Employee Engagement"
            )

        else:

            risk = "High Risk"

            recommendation = (
                "Immediate HR Intervention"
            )

        # Display Results
        st.success(
            f"Risk Level: {risk}"
        )

        st.info(
            f"Risk Probability: {round(probability, 2)}"
        )

        st.warning(
            f"Recommendation: {recommendation}"
        )

# ==========================================================
# UPLOAD DATASET
# ==========================================================

elif menu == "Upload Dataset":

    st.subheader("📂 Upload Employee Dataset")

    uploaded_file = st.file_uploader(

        "Upload CSV File",

        type=['csv']
    )

    if uploaded_file is not None:

        df = pd.read_csv(
            uploaded_file
        )

        st.write(df.head())

        st.write(
            "Dataset Shape:",
            df.shape
        )

        st.write(
            "Missing Values"
        )

        st.write(
            df.isnull().sum()
        )

        # Prediction
        predictions = model.predict(df)

        probabilities = model.predict_proba(
            df
        )[:,1]

        df['Prediction'] = predictions

        df['Risk Probability'] = probabilities

        st.subheader("Prediction Results")

        st.write(df.head())

        # Download Button
        csv = df.to_csv(index=False)

        st.download_button(

            label="Download Results",

            data=csv,

            file_name='employee_predictions.csv',

            mime='text/csv'
        )

# ==========================================================
# ANALYTICS
# ==========================================================

elif menu == "Analytics":

    st.subheader("📊 Advanced HR Analytics")

    # Monthly Trend
    trend_data = pd.DataFrame({

        'Month': [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun'
        ],

        'Attrition': [
            20,
            35,
            28,
            40,
            32,
            25
        ]
    })

    fig3 = px.line(

        trend_data,

        x='Month',

        y='Attrition',

        markers=True,

        title='Monthly Attrition Trend'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # Heatmap
    heat_data = np.random.rand(5, 5)

    fig4 = go.Figure(

        data=go.Heatmap(
            z=heat_data
        )
    )

    fig4.update_layout(
        title='Employee Performance Heatmap'
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    "Developed using Streamlit + Machine Learning"
)