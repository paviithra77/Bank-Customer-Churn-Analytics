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
# Paths and page configuration (must be the first Streamlit call)
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
# Constants
# --------------------------------------------------
FEATURE_COLUMNS = [
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

AGE_BINS = [0, 30, 40, 50, 60, 120]
AGE_LABELS = ["18–30", "31–40", "41–50", "51–60", "61+"]

RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
PRIMARY = "#1769AA"
ACCENT = "#168C9B"


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #F5F8FD 0%, #E8F1FB 100%);
        animation: fadeIn 0.6s ease-out;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1400px;
        animation: fadeInUp 0.5s ease-out;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Headings */
    h1, h2, h3, h4 {
        color: #142D4E;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    h2 a, h3 a { display: none; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF, #EEF5FF);
        border: 1px solid #DCE7F5;
        border-left: 5px solid #1769AA;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(25, 55, 95, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease,
                    border-color 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 26px rgba(25, 75, 130, 0.14);
        border-color: #7CB9F2;
        border-left-color: #168C9B;
    }
    [data-testid="stMetricLabel"] {
        color: #536781;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        color: #1769AA;
        font-weight: 800;
    }

    /* Bordered containers act as cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 14px;
        border-color: #DCE7F5;
        box-shadow: 0 3px 12px rgba(25, 55, 95, 0.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #DCE7F5;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 14px;
        font-weight: 600;
        color: #536781;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(100deg, #1769AA, #168C9B);
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        background: linear-gradient(100deg, #1769AA, #168C9B);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.3rem;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(100deg, #124F80, #087C8A);
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(23, 105, 170, 0.28);
        border: none;
    }
    .stButton > button:active,
    .stFormSubmitButton > button:active,
    .stDownloadButton > button:active { transform: scale(0.97); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #142D4E, #0B1C32);
    }
    [data-testid="stSidebar"] * { color: #F4F8FF; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 8px 12px;
        border-radius: 10px;
        transition: background 0.2s ease, transform 0.2s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: translateX(4px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(124, 185, 242, 0.25);
    }

    /* Inputs */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div {
        border-radius: 9px;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        box-shadow: 0 0 0 2px rgba(23, 105, 170, 0.2);
    }

    /* Tables / charts */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #DCE7F5;
    }

    hr { border-color: #D5E1F0; }

    /* Page header (non-home pages) */
    .page-header {
        background: linear-gradient(120deg, #142D4E, #1769AA, #168C9B);
        padding: 22px 28px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 8px 22px rgba(20, 45, 78, 0.15);
    }
    .page-header .eyebrow {
        color: #BDE9FF; font-size: 12px; font-weight: 700;
        letter-spacing: 2px; margin: 0;
    }
    .page-header h1 {
        color: #FFFFFF !important; font-size: 28px; margin: 4px 0 2px 0;
    }
    .page-header p { color: #E4F2FF; margin: 0; font-size: 15px; }

    /* Home hero */
    .hero {
        background: linear-gradient(120deg, #102642, #1769AA, #168C9B);
        padding: 44px 40px;
        border-radius: 22px;
        box-shadow: 0 14px 32px rgba(20, 45, 78, 0.2);
    }
    .hero .eyebrow {
        color: #BDE7FF; font-size: 13px; font-weight: 700;
        letter-spacing: 3px; margin: 0;
    }
    .hero h1 {
        color: #FFFFFF !important; font-size: 42px; font-weight: 800;
        margin: 10px 0 14px 0;
    }
    .hero p { color: #F0F7FF; font-size: 18px; line-height: 1.7; margin: 0; }

    .footer {
        text-align: center; color: #64748B; padding: 10px;
        font-size: 14px;
    }
    .footer p { margin: 2px 0; }

    /* Accessibility: respect reduced-motion settings */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    /* Mobile layout */
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
        [data-testid="stMetric"] { padding: 12px; }
        .hero { padding: 28px 22px; }
        .hero h1 { font-size: 30px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def show_df(data, **kwargs):
    """Display a dataframe full-width across Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <p class="eyebrow">BANKING INTELLIGENCE</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_age_group(data):
    data = data.copy()
    data["Age Group"] = pd.cut(
        data["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    return data


def risk_label_from(probability):
    if probability >= 0.60:
        return "High Risk"
    if probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data(show_spinner="Scoring customers...")
def compute_risk_counts(_model, features):
    """Score every customer once and cache the result."""
    probabilities = _model.predict_proba(features)[:, 1]
    categories = pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=RISK_LABELS,
    )
    return (
        pd.Series(categories)
        .value_counts()
        .reindex(RISK_LABELS, fill_value=0)
    )


# --------------------------------------------------
# Load data and trained model
# --------------------------------------------------
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
except Exception as error:  # corrupt pickle, version mismatch, etc.
    st.error(f"Could not load the data or the model: {error}")
    st.stop()

missing_columns = [
    column for column in FEATURE_COLUMNS + ["Exited"] if column not in df.columns
]
if missing_columns:
    st.error(f"The dataset is missing required columns: {missing_columns}")
    st.stop()


# --------------------------------------------------
# Shared dataset summary calculations
# --------------------------------------------------
total = len(df)
churned = int(df["Exited"].sum())
retained = total - churned
churn_rate = (churned / total * 100) if total else 0.0
retained_rate = (retained / total * 100) if total else 0.0


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🏦 Bank Insights")
    st.caption("Customer Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Overview",
            "🔍 Churn Analysis",
            "🤖 Predict Churn",
            "📈 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Bank Customer Churn Analytics")
    st.caption("Machine Learning Dashboard")


# --------------------------------------------------
# Home page
# --------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">BANKING INTELLIGENCE</p>
            <h1>Customer Churn Analytics</h1>
            <p>
                Understand customer behavior. Identify churn risks.
                Make data-driven retention decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.subheader("About the Application")
    st.write(
        "Bank Customer Churn Analytics is a machine learning-based "
        "application that analyses customer information to identify "
        "churn patterns and estimate the likelihood of a customer "
        "leaving the bank. It helps users understand customer behavior "
        "and make informed retention decisions."
    )

    st.divider()
    st.subheader("Explore the Platform")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📊 Customer Insights")
            st.write(
                "Explore customer statistics, churn rates, "
                "retention trends, and key business insights."
            )
        with st.container(border=True):
            st.markdown("### 🔍 Churn Analysis")
            st.write(
                "Discover churn patterns across age groups, "
                "geographical locations, and activity levels."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### 🤖 Churn Prediction")
            st.write(
                "Use the trained machine learning model to estimate "
                "an individual customer's risk of leaving the bank."
            )
        with st.container(border=True):
            st.markdown("### 📈 Model Performance")
            st.write(
                "Review model accuracy, precision, recall, "
                "and confusion matrix results."
            )

    st.divider()
    st.subheader("Getting Started")
    st.markdown(
        """
        **Step 1:** Select **Overview** to explore customer metrics.

        **Step 2:** Select **Churn Analysis** to study churn patterns.

        **Step 3:** Select **Predict Churn** to assess an individual customer.

        **Step 4:** Select **Model Performance** to evaluate the model.
        """
    )
    st.info(
        "Select a page from the navigation menu in the sidebar "
        "to explore the application."
    )


# --------------------------------------------------
# Overview page
# --------------------------------------------------
elif page == "📊 Overview":
    page_header(
        "Customer Overview",
        "Key metrics, risk levels, segments and churn drivers at a glance.",
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Customers", f"{total:,}")
    metric2.metric("Customers Churned", f"{churned:,}")
    metric3.metric("Customers Retained", f"{retained:,}")
    metric4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.write("")

    (
        tab_summary,
        tab_risk,
        tab_search,
        tab_records,
        tab_segments,
        tab_drivers,
        tab_actions,
    ) = st.tabs(
        [
            "📌 Summary",
            "🎯 Risk Distribution",
            "🔎 Search Customer",
            "📋 Records",
            "👥 Segmentation",
            "📊 Churn Drivers",
            "📝 Action Planner",
        ]
    )

    # ---- Summary ----
    with tab_summary:
        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.subheader("Customer Churn Distribution")
                chart_data = pd.DataFrame(
                    {
                        "Customer Type": ["Retained", "Churned"],
                        "Customers": [retained, churned],
                    }
                ).set_index("Customer Type")
                st.bar_chart(chart_data, color=PRIMARY)

        with right:
            with st.container(border=True):
                st.subheader("Churn Rate by Geography")
                geography_data = (
                    df.groupby("Geography")["Exited"]
                    .mean()
                    .mul(100)
                    .sort_values(ascending=False)
                )
                st.bar_chart(geography_data, color=ACCENT)
                st.caption("Percentage of customers who left in each geography.")

        st.subheader("Quick Insights")
        insight1, insight2 = st.columns(2)
        insight1.info(
            f"{churn_rate:.1f}% of customers in this dataset have left the bank."
        )
        insight2.info(
            f"{retained:,} customers are currently classified as retained "
            f"({retained_rate:.1f}% of the dataset)."
        )

    # ---- Risk distribution ----
    with tab_risk:
        st.subheader("🎯 Churn Risk Distribution")
        try:
            risk_counts = compute_risk_counts(model, df[FEATURE_COLUMNS].copy())

            r1, r2, r3 = st.columns(3)
            r1.metric("🟢 Low Risk", f"{int(risk_counts['Low Risk']):,}")
            r2.metric("🟡 Medium Risk", f"{int(risk_counts['Medium Risk']):,}")
            r3.metric("🔴 High Risk", f"{int(risk_counts['High Risk']):,}")

            with st.container(border=True):
                st.bar_chart(risk_counts, color="#E67E22")
            st.caption(
                "Risk categories use estimated churn probabilities: "
                "Low below 30%, Medium 30–60%, High above 60%."
            )
        except Exception as error:
            st.error(
                "Could not score customers with the saved model. "
                f"Details: {error}"
            )

    # ---- Search customer ----
    with tab_search:
        st.subheader("🔎 Search Customer")

        if "CustomerId" not in df.columns:
            st.warning("The dataset has no CustomerId column.")
        else:
            with st.form("customer_search_form"):
                customer_id = st.number_input(
                    "Enter Customer ID",
                    min_value=0,
                    step=1,
                    value=int(df["CustomerId"].iloc[0]),
                )
                search_clicked = st.form_submit_button("Search Customer")

            if search_clicked:
                customer_result = df[df["CustomerId"] == int(customer_id)]

                if not customer_result.empty:
                    customer_row = customer_result.iloc[0]
                    st.success("Customer found!")

                    detail1, detail2, detail3 = st.columns(3)
                    detail1.metric("Age", int(customer_row["Age"]))
                    detail2.metric(
                        "Credit Score", int(customer_row["CreditScore"])
                    )
                    detail3.metric(
                        "Account Status",
                        "Churned" if customer_row["Exited"] == 1 else "Retained",
                    )

                    with st.container(border=True):
                        st.write("**Geography:**", customer_row["Geography"])
                        st.write("**Gender:**", customer_row["Gender"])
                        st.write("**Balance:**", f'{customer_row["Balance"]:,.2f}')
                        st.write("**Tenure:**", int(customer_row["Tenure"]), "years")
                else:
                    st.warning("Customer ID not found. Please try another ID.")

    # ---- Records explorer ----
    with tab_records:
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
            column
            for column in [
                "CustomerId",
                "Geography",
                "Gender",
                "Age",
                "CreditScore",
                "Balance",
                "IsActiveMember",
                "Exited",
            ]
            if column in display_df.columns
        ]

        show_df(
            display_df[display_columns].rename(
                columns={
                    "CustomerId": "Customer ID",
                    "CreditScore": "Credit Score",
                    "IsActiveMember": "Active Member (1=Yes)",
                    "Exited": "Churned (1=Yes)",
                }
            ),
            hide_index=True,
        )
        st.caption(f"Showing {len(display_df):,} customer records.")

        st.download_button(
            label="📥 Download Filtered Customer Data",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customer_records.csv",
            mime="text/csv",
        )

    # ---- Segmentation ----
    with tab_segments:
        st.subheader("👥 Customer Segmentation")

        segment_option = st.selectbox(
            "Choose a segmentation method",
            ["Age Group", "Account Balance", "Activity Status"],
        )

        segment_df = df.copy()

        if segment_option == "Age Group":
            segment_df["Customer Segment"] = pd.cut(
                segment_df["Age"],
                bins=AGE_BINS,
                labels=AGE_LABELS,
                include_lowest=True,
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
            segment_df["Customer Segment"] = segment_df["IsActiveMember"].map(
                {0: "Inactive", 1: "Active"}
            )

        segment_summary = segment_df.groupby(
            "Customer Segment", observed=False
        ).agg(
            Customers=("Exited", "size"),
            Churned=("Exited", "sum"),
            Churn_Rate=("Exited", "mean"),
        )
        segment_summary["Churn Rate (%)"] = segment_summary.pop("Churn_Rate") * 100

        seg_left, seg_right = st.columns([1, 1])
        with seg_left:
            show_df(segment_summary.round(2))
        with seg_right:
            st.bar_chart(segment_summary["Churn Rate (%)"], color=PRIMARY)

        st.caption(
            "Compare customer group sizes and churn rates to identify segments "
            "that may benefit from further investigation."
        )

    # ---- Churn drivers ----
    with tab_drivers:
        st.subheader("📊 Churn Drivers Analysis")

        driver_df = add_age_group(df)
        driver_df["Activity Status"] = driver_df["IsActiveMember"].map(
            {0: "Inactive", 1: "Active"}
        )
        driver_df["Credit Card Status"] = driver_df["HasCrCard"].map(
            {0: "No Credit Card", 1: "Has Credit Card"}
        )

        driver_options = {
            "Age Group": "Age Group",
            "Geography": "Geography",
            "Gender": "Gender",
            "Activity Status": "Activity Status",
            "Number of Products": "NumOfProducts",
            "Credit Card Status": "Credit Card Status",
        }

        driver_choice = st.selectbox(
            "Analyze churn by",
            list(driver_options.keys()),
            key="driver_analysis_choice",
        )
        driver_column = driver_options[driver_choice]

        driver_data = (
            driver_df.groupby(driver_column, observed=False)["Exited"]
            .agg(["mean", "count"])
            .reset_index()
        )
        driver_data["Churn Rate (%)"] = driver_data["mean"] * 100
        driver_data = driver_data.rename(columns={"count": "Customer Count"})
        driver_data[driver_column] = driver_data[driver_column].astype(str)

        drv_left, drv_right = st.columns(2)
        with drv_left:
            st.bar_chart(
                driver_data.set_index(driver_column)["Churn Rate (%)"],
                color=ACCENT,
            )
        with drv_right:
            show_df(
                driver_data.drop(columns=["mean"]).round(2),
                hide_index=True,
            )

        st.caption(
            "These relationships describe the dataset and do not prove that "
            "a characteristic causes a customer to leave."
        )

    # ---- Retention action planner ----
    with tab_actions:
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
            "Actual retention actions should follow bank policies and "
            "customer needs."
        )


# --------------------------------------------------
# Churn analysis page
# --------------------------------------------------
elif page == "🔍 Churn Analysis":
    page_header(
        "Customer Churn Insights",
        "Explore how churn varies across customer groups.",
    )

    age_df = add_age_group(df)

    age_churn = (
        age_df.groupby("Age Group", observed=False)["Exited"]
        .mean()
        .mul(100)
        .dropna()
    )

    activity_churn = df.groupby("IsActiveMember")["Exited"].mean().mul(100)
    activity_churn.index = activity_churn.index.map(
        {0: "Inactive Members", 1: "Active Members"}
    )

    geography_churn = (
        df.groupby("Geography")["Exited"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Churn Rate by Age Group")
            st.bar_chart(age_churn, color=PRIMARY)
            st.caption("Percentage of customers who left in each age group.")

    with col2:
        with st.container(border=True):
            st.subheader("Churn Rate by Activity")
            st.bar_chart(activity_churn, color=ACCENT)
            st.caption("Compares churn rates for active and inactive customers.")

    with st.container(border=True):
        st.subheader("Churn Rate by Geography")
        st.bar_chart(geography_churn, color=PRIMARY)
        st.caption("Geographies are ordered from highest to lowest churn rate.")

    st.divider()
    st.subheader("📥 Download Churn Analysis Report")

    report_df = df.copy()
    report_df["Churn Status"] = report_df["Exited"].map(
        {0: "Retained", 1: "Churned"}
    )

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
    page_header(
        "Predict Customer Churn",
        "Enter customer details to estimate their churn risk.",
    )

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
            age = st.number_input("Age", min_value=18, max_value=100, value=42)

        with col2:
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=10, value=5
            )
            balance = st.number_input("Balance", min_value=0.0, value=80000.0)
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
        )[FEATURE_COLUMNS]

        try:
            prediction = model.predict(customer)[0]
            probability = float(model.predict_proba(customer)[0][1])

            st.markdown("### Customer Risk Summary")

            risk_label = risk_label_from(probability)

            summary1, summary2, summary3 = st.columns(3)
            summary1.metric("Risk Category", risk_label)
            summary2.metric("Churn Probability", f"{probability:.1%}")
            summary3.metric(
                "Suggested Action",
                "Priority Review"
                if probability >= 0.60
                else "Monitor"
                if probability >= 0.30
                else "Regular Follow-up",
            )

            st.divider()
            st.subheader("Prediction Result")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Estimated Churn Probability", f"{probability:.2%}")
                st.progress(min(max(probability, 0.0), 1.0))

            with result_col2:
                if probability >= 0.60:
                    st.error("🔴 High Churn Risk")
                    st.write("Consider prioritizing this customer for retention.")
                elif probability >= 0.30:
                    st.warning("🟡 Medium Churn Risk")
                    st.write("Consider monitoring this customer's engagement.")
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
                    "Explore whether another suitable banking product meets "
                    "the customer's needs."
                )
            if age >= 50:
                recommendations.append(
                    "Offer appropriate support and services based on "
                    "customer preferences."
                )
            if not recommendations:
                recommendations.append(
                    "Continue regular engagement and monitor customer "
                    "satisfaction."
                )

            with st.container(border=True):
                for recommendation in recommendations:
                    st.markdown(f"- {recommendation}")

        except Exception as error:
            st.error(
                "The model could not process these inputs. "
                f"Please check the model's expected features. Details: {error}"
            )

    st.caption("Predictions are estimates, not guarantees of future behavior.")


# --------------------------------------------------
# Model performance page
# --------------------------------------------------
elif page == "📈 Model Performance":
    page_header(
        "Machine Learning Model Performance",
        "Evaluate how the trained model performs on a held-out test set.",
    )

    X = df[FEATURE_COLUMNS]
    y = df["Exited"]

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

        st.write("")
        with st.container(border=True):
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Stayed", "Actual: Left"],
                columns=["Predicted: Stayed", "Predicted: Left"],
            )
            show_df(cm_df)

            st.caption(
                "Evaluation uses a stratified 80/20 test split with "
                "random_state=42. Churn recall measures the proportion of "
                "actual churners correctly identified by the model."
            )

        if recall < 0.6:
            st.warning(
                "The model misses some customers who actually leave. "
                "Consider improving churn recall before relying on "
                "predictions for business decisions."
            )

    except Exception as error:
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
    <div class="footer">
        <p><strong>Bank Customer Churn Analytics</strong></p>
        <p>Customer Insights | Machine Learning | Retention Analytics</p>
        <p>Developed as a data analytics and machine learning project</p>
    </div>
    """,
    unsafe_allow_html=True,
)
