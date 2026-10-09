import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths and page configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

st.set_page_config(
    page_title="Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


# --------------------------------------------------
# Custom dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FC;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }
    h1, h2, h3 {
        color: #142D4E;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF, #EDF5FF);
        border: 1px solid #DCE7F5;
        padding: 20px 22px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(25, 55, 95, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #536781;
        font-size: 14px;
    }
    [data-testid="stMetricValue"] {
        color: #1769AA;
        font-weight: 700;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #1769AA, #168C9B);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #124F80;
        color: white;
        border: none;
    }
    [data-testid="stSidebar"] {
        background: #142D4E;
    }
    [data-testid="stSidebar"] * {
        color: #F4F8FF;
    }
    hr {
        border-color: #DCE5F0;
    }
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🏦 Bank Insights")
    st.caption("Customer Intelligence Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "📊 Overview",
            "🔍 Churn Analysis",
            "🤖 Predict Churn",
            "📈 Model Performance",
        ],
    )

    st.divider()
    st.caption("Bank Customer Churn Analytics")
    st.caption("Machine Learning Dashboard")


# --------------------------------------------------
# Hero banner
# --------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #142D4E, #1769AA, #168C9B);
        padding: 32px;
        border-radius: 18px;
        margin-bottom: 26px;
    ">
        <div style="
            color: #BDE9FF;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
        ">
            BANKING INTELLIGENCE
        </div>
        <h1 style="
            color: white !important;
            font-size: 34px;
            margin: 10px 0;
        ">
            Customer Churn Analytics
        </h1>
        <p style="
            color: #E4F2FF;
            font-size: 16px;
            margin-bottom: 0;
        ">
            Understand customer behavior. Identify churn risks.
            Make data-driven retention decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load data and trained model
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


try:
    df = load_data()
    model = load_model()
except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.info(
        "Check that Churn_Modelling.csv is inside the data folder "
        "and churn_model.pkl is inside the model folder."
    )
    st.stop()


# --------------------------------------------------
# Shared dataset summary calculations
# Keep these outside page conditions so every page can use them.
# --------------------------------------------------
total = len(df)
churned = int(df["Exited"].sum())
retained = total - churned
churn_rate = (churned / total * 100) if total else 0.0


# --------------------------------------------------
# Overview page
# --------------------------------------------------
if page == "📊 Overview":
    st.subheader("Customer Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Customers", f"{total:,}")
    metric2.metric("Customers Churned", f"{churned:,}")
    metric3.metric("Customers Retained", f"{retained:,}")
    metric4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")
        chart_data = pd.DataFrame(
            {
                "Customer Type": ["Retained", "Churned"],
                "Customers": [retained, churned],
            }
        ).set_index("Customer Type")
        st.bar_chart(chart_data)

    with right:
        st.subheader("Churn Rate by Geography")
        geography_data = (
            df.groupby("Geography")["Exited"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        st.bar_chart(geography_data)
        st.caption("Percentage of customers who left in each geography.")

    st.divider()
    st.subheader("Quick Insights")
    
    

    insight1, insight2 = st.columns(2)

    insight1.info(
        f"{churn_rate:.1f}% of customers in this dataset have left the bank."
    )

    insight2.info(
        f"{retained:,} customers are currently classified as retained "
        f"({(retained / total * 100) if total else 0:.1f}% of the dataset)."
    )
    
    st.divider()
    st.subheader("🎯 Churn Risk Distribution")

    feature_columns = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    risk_df = df[feature_columns].copy()
    risk_probabilities = model.predict_proba(risk_df)[:, 1]

    risk_categories = pd.cut(
        risk_probabilities,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    risk_counts = (
        risk_categories.value_counts()
        .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    )

    st.bar_chart(risk_counts)
    st.caption(
        "Risk categories use estimated churn probabilities: "
        "Low below 30%, Medium 30–60%, High above 60%."
    )
    
    st.divider()
    st.subheader("🔎 Search Customer")

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=1,
        step=1,
        value=1,
    )

    if st.button("Search Customer"):
        customer_result = df[df["CustomerId"] == customer_id]

        if not customer_result.empty:
            customer_row = customer_result.iloc[0]

            st.success("Customer found!")

            detail1, detail2, detail3 = st.columns(3)

            detail1.metric("Age", int(customer_row["Age"]))
            detail2.metric("Credit Score", int(customer_row["CreditScore"]))
            detail3.metric(
                "Account Status",
                "Churned" if customer_row["Exited"] == 1 else "Retained",
            )

            st.write("**Geography:**", customer_row["Geography"])
            st.write("**Gender:**", customer_row["Gender"])
            st.write("**Balance:**", f'{customer_row["Balance"]:,.2f}')
            st.write("**Tenure:**", int(customer_row["Tenure"]), "years")
        else:
            st.warning("Customer ID not found. Please try another ID.")
            
    st.divider()
    st.subheader("📋 Customer Records Explorer")

    status_filter = st.selectbox(
        "Filter by Customer Status",
        ["All Customers", "Retained", "Churned"],
    )

    display_df = df.copy()

    if status_filter == "Retained":
        display_df = display_df[display_df["Exited"] == 0]
    elif status_filter == "Churned":
        display_df = display_df[display_df["Exited"] == 1]

    display_columns = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "CreditScore",
        "Balance",
        "IsActiveMember",
        "Exited",
    ]

    st.dataframe(
        display_df[display_columns].rename(
            columns={
                "CustomerId": "Customer ID",
                "Geography": "Geography",
                "Gender": "Gender",
                "Age": "Age",
                "CreditScore": "Credit Score",
                "Balance": "Balance",
                "IsActiveMember": "Active Member (1=Yes)",
                "Exited": "Churned (1=Yes)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {len(display_df):,} customer records.")
    
    st.download_button(
        label="📥 Download Filtered Customer Data",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_records.csv",
        mime="text/csv",
    )
    
    st.divider()
    st.subheader("👥 Customer Segmentation")

    segment_option = st.selectbox(
        "Choose a segmentation method",
        ["Age Group", "Account Balance", "Activity Status"],
    )

    segment_df = df.copy()

    if segment_option == "Age Group":
        segment_df["Customer Segment"] = pd.cut(
            segment_df["Age"],
            bins=[0, 30, 40, 50, 60, 120],
            labels=["18–30", "31–40", "41–50", "51–60", "61+"],
        )

    elif segment_option == "Account Balance":
        segment_df["Customer Segment"] = pd.cut(
            segment_df["Balance"],
            bins=[-1, 0, 50000, 100000, float("inf")],
            labels=[
                "Zero Balance",
                "Low Balance",
                "Medium Balance",
                "High Balance",
            ],
        )

    else:
        segment_df["Customer Segment"] = segment_df[
            "IsActiveMember"
        ].map({
            0: "Inactive",
            1: "Active",
        })

    segment_summary = (
        segment_df.groupby("Customer Segment", observed=False)
        .agg(
            Customers=("Exited", "size"),
            Churned=("Exited", "sum"),
            Churn_Rate=("Exited", "mean"),
        )
    )

    segment_summary["Churn Rate (%)"] = (
        segment_summary.pop("Churn_Rate") * 100
    )

    st.dataframe(segment_summary.round(2), use_container_width=True)
    st.bar_chart(segment_summary["Churn Rate (%)"])

    st.caption(
        "Compare customer group sizes and churn rates to identify segments "
        "that may benefit from further investigation."
    )
    
    st.divider()
    st.subheader("📊 Churn Drivers Analysis")

    driver_options = {
        "Age Group": "Age",
        "Geography": "Geography",
        "Gender": "Gender",
        "Activity Status": "IsActiveMember",
        "Number of Products": "NumOfProducts",
        "Credit Card Status": "HasCrCard",
    }

    driver_choice = st.selectbox(
        "Analyze churn by",
        list(driver_options.keys()),
        key="driver_analysis_choice",
    )

    driver_column = driver_options[driver_choice]

    driver_data = (
        df.groupby(driver_column, observed=False)["Exited"]
        .agg(["mean", "count"])
        .reset_index()
    )

    driver_data["Churn Rate (%)"] = driver_data["mean"] * 100
    driver_data = driver_data.rename(
        columns={"count": "Customer Count"}
    )

    st.bar_chart(
        driver_data.set_index(driver_column)["Churn Rate (%)"]
    )

    st.dataframe(
        driver_data.drop(columns=["mean"]).round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These relationships describe the dataset and do not prove that "
        "a characteristic causes a customer to leave."
    )
    
    st.divider()
    st.subheader("📝 Customer Retention Action Planner")

    selected_risk = st.selectbox(
        "Select customer risk level",
        ["High Risk", "Medium Risk", "Low Risk"],
        key="retention_action_risk",
    )

    if selected_risk == "High Risk":
        st.error("Priority: High")
        st.markdown(
            """
            - Contact the customer to understand their concerns.
            - Review recent complaints and service experience.
            - Consider a suitable personalized retention offer.
            - Follow up and monitor the customer's response.
            """
        )

    elif selected_risk == "Medium Risk":
        st.warning("Priority: Medium")
        st.markdown(
            """
            - Monitor changes in customer activity.
            - Share relevant banking products and services.
            - Check customer satisfaction through appropriate follow-up.
            """
        )

    else:
        st.success("Priority: Routine")
        st.markdown(
            """
            - Maintain regular customer communication.
            - Continue monitoring satisfaction and engagement.
            - Offer relevant services when appropriate.
            """
        )

    st.caption(
        "These are general suggestions for a project demonstration. "
        "Actual retention actions should follow bank policies and customer needs."
    )

# --------------------------------------------------
# Churn analysis page
# --------------------------------------------------
elif page == "🔍 Churn Analysis":
    st.header("📊 Customer Churn Insights")
    st.write("Explore how churn varies across customer groups.")

    age_df = df.copy()
    age_df["Age Group"] = pd.cut(
        age_df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18–30", "31–40", "41–50", "51–60", "61+"],
        include_lowest=True,
    )

    age_churn = (
        age_df.groupby("Age Group", observed=False)["Exited"]
        .mean()
        .mul(100)
        .dropna()
    )

    activity_churn = (
        df.groupby("IsActiveMember")["Exited"]
        .mean()
        .mul(100)
    )
    activity_churn.index = activity_churn.index.map(
        {
            0: "Inactive Members",
            1: "Active Members",
        }
    )

    geography_churn = (
        df.groupby("Geography")["Exited"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Age Group")
        st.bar_chart(age_churn)
        st.caption("Percentage of customers who left in each age group.")

    with col2:
        st.subheader("Churn Rate by Activity")
        st.bar_chart(activity_churn)
        st.caption("Compares churn rates for active and inactive customers.")

    st.divider()
    st.subheader("Churn Rate by Geography")
    st.bar_chart(geography_churn)
    st.caption("Geographies are ordered from highest to lowest churn rate.")
    
    st.divider()
    st.subheader("📥 Download Churn Analysis Report")

    report_df = df.copy()
    report_df["Churn Status"] = report_df["Exited"].map({
        0: "Retained",
        1: "Churned"
    })

    st.download_button(
        label="Download Customer Report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="bank_customer_churn_report.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# Customer prediction page
# --------------------------------------------------
elif page == "🤖 Predict Churn":
    st.header("🔍 Predict Customer Churn")
    st.write("Enter customer details to estimate their churn risk.")

    with st.form("customer_prediction"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=850, value=650
            )
            geography = st.selectbox(
                "Geography", sorted(df["Geography"].dropna().unique())
            )
            gender = st.selectbox(
                "Gender", sorted(df["Gender"].dropna().unique())
            )
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=42
            )

        with col2:
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=10, value=5
            )
            balance = st.number_input(
                "Balance", min_value=0.0, value=80000.0
            )
            products = st.number_input(
                "Number of Products", min_value=1, max_value=4, value=2
            )

        with col3:
            credit_card = st.selectbox(
                "Has Credit Card",
                [1, 0],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            active_member = st.selectbox(
                "Active Member",
                [1, 0],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            salary = st.number_input(
                "Estimated Salary", min_value=0.0, value=60000.0
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": products,
                    "HasCrCard": credit_card,
                    "IsActiveMember": active_member,
                    "EstimatedSalary": salary,
                }
            ]
        )

        try:
            prediction = model.predict(customer)[0]
            probability = model.predict_proba(customer)[0][1]
            
            st.markdown("### Customer Risk Summary")

            risk_label = (
                "High Risk" if probability >= 0.60
                else "Medium Risk" if probability >= 0.30
                else "Low Risk"
            )

            summary1, summary2, summary3 = st.columns(3)

            summary1.metric("Risk Category", risk_label)
            summary2.metric("Churn Probability", f"{probability:.1%}")
            summary3.metric(
                "Suggested Action",
                "Priority Review" if probability >= 0.60
                else "Monitor" if probability >= 0.30
                else "Regular Follow-up"
            )

            st.divider()
            st.subheader("Prediction Result")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.2%}"
                )
                st.progress(float(probability))

            with result_col2:
                if probability >= 0.60:
                    st.error("🔴 High Churn Risk")
                    st.write(
                        "Consider prioritizing this customer for retention."
                    )
                elif probability >= 0.30:
                    st.warning("🟡 Medium Churn Risk")
                    st.write(
                        "Consider monitoring this customer's engagement."
                    )
                else:
                    st.success("🟢 Low Churn Risk")
                    st.write(
                        "The estimated churn probability is relatively low."
                    )

            if prediction == 1:
                st.info("The model predicts that this customer is likely to leave.")
            else:
                st.info("The model predicts that this customer is likely to stay.")
                
            st.subheader("💡 Customer Retention Recommendations")

            recommendations = []

            if probability >= 0.60:
                recommendations.append(
                    "Contact the customer with a personalized retention offer."
                )
                recommendations.append(
                    "Review customer complaints and service experience."
                )

            if active_member == 0:
                recommendations.append(
                    "Encourage the customer to use banking services regularly."
                )

            if products == 1:
                recommendations.append(
                    "Explore whether another suitable banking product meets the customer's needs."
                )

            if age >= 50:
                recommendations.append(
                    "Offer appropriate support and services based on customer preferences."
                )

            if not recommendations:
                recommendations.append(
                    "Continue regular engagement and monitor customer satisfaction."
                )

            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")

        except (ValueError, KeyError) as error:
            st.error(
                "The model could not process these inputs. "
                f"Please check the model's expected features. Details: {error}"
            )

    st.caption(
        "Predictions are estimates, not guarantees of future behavior."
    )


# --------------------------------------------------
# Model performance page
# --------------------------------------------------
elif page == "📈 Model Performance":
    st.header("🤖 Machine Learning Model Performance")
    st.write("Evaluate how the trained model performs on a held-out test set.")

    evaluation_df = df.drop(
        columns=["RowNumber", "CustomerId", "Surname"],
        errors="ignore",
    )
    X = evaluation_df.drop(columns=["Exited"])
    y = evaluation_df["Exited"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    try:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Accuracy", f"{accuracy:.2%}")
        metric2.metric("Churn Precision", f"{precision:.2%}")
        metric3.metric("Churn Recall", f"{recall:.2%}")
        metric4.metric("Churn F1-Score", f"{f1:.2%}")

        st.divider()
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Stayed", "Actual: Left"],
            columns=["Predicted: Stayed", "Predicted: Left"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.caption(
            "Evaluation uses a stratified 80/20 test split with random_state=42. "
            "Churn recall measures the proportion of actual churners correctly "
            "identified by the model."
        )

        if recall < 0.6:
            st.warning(
                "The model misses some customers who actually leave. "
                "Consider improving churn recall before relying on predictions "
                "for business decisions."
            )

    except (ValueError, KeyError) as error:
        st.error(
            "Model evaluation failed. Check that the saved model was trained "
            f"with the expected features. Details: {error}"
        )
        
# --------------------------------------------------
# Dashboard footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #64748b; padding: 12px;">
        <p><strong>Bank Customer Churn Analytics</strong></p>
        <p>Customer Insights | Machine Learning | Retention Analytics</p>
        <p>Developed as a data analytics and machine learning project</p>
    </div>
    """,
    unsafe_allow_html=True,
)
