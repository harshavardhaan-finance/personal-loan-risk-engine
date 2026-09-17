import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Personal Loan Risk Engine",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("personal_loan_pd_model.pkl")


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💳 Personal Loan Credit Risk Engine")

st.markdown(
    """
    **Machine Learning Based Personal Loan Risk Assessment & Pricing**
    
    Enter applicant information below to estimate:
    - Probability of Default (PD)
    - Expected Loss
    - Risk Band
    - Indicative Interest Rate
    """
)


st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=0.0,
        value=800000.0,
        step=25000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=750.0,
        step=1.0
    )

    existing_emi = st.number_input(
        "Existing Monthly EMI (₹)",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )

    net_worth = st.number_input(
        "Net Worth (₹)",
        min_value=0.0,
        value=1000000.0,
        step=25000.0
    )


with col2:

    credit_utilization = st.number_input(
        "Credit Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    dpd_30_count = st.number_input(
        "30+ DPD Count",
        min_value=0,
        value=0,
        step=1
    )

    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=0.0,
        value=500000.0,
        step=25000.0
    )


st.divider()


# --------------------------------------------------
# CREDIT RISK ASSESSMENT
# --------------------------------------------------

if st.button(
    "🔍 ASSESS CREDIT RISK",
    type="primary",
    use_container_width=True
):

    # Create applicant dataframe
    applicant = pd.DataFrame({
        "annual_income": [annual_income],
        "credit_score": [credit_score],
        "existing_emi": [existing_emi],
        "net_worth": [net_worth],
        "credit_utilization": [credit_utilization],
        "dpd_30_count": [dpd_30_count],
        "loan_amount": [loan_amount]
    })

    # Model prediction
    pd_probability = model.predict_proba(applicant)[0][1]

    # Convert to percentage
    pd_percentage = pd_probability * 100


    # --------------------------------------------------
    # LGD & EAD
    # --------------------------------------------------

    LGD = 0.60

    EAD = loan_amount


    # Expected Loss
    expected_loss = pd_probability * LGD * EAD


    # --------------------------------------------------
    # RISK BAND
    # --------------------------------------------------

    if pd_probability < 0.02:

        risk_band = "LOW"
        indicative_rate = 11.0

    elif pd_probability < 0.05:

        risk_band = "MODERATE"
        indicative_rate = 13.0

    elif pd_probability < 0.10:

        risk_band = "HIGH"
        indicative_rate = 16.0

    else:

        risk_band = "VERY HIGH"
        indicative_rate = 20.0


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader("📊 Credit Risk Assessment")


    r1, r2, r3, r4 = st.columns(4)


    with r1:

        st.metric(
            "Probability of Default",
            f"{pd_percentage:.2f}%"
        )


    with r2:

        st.metric(
            "Risk Band",
            risk_band
        )


    with r3:

        st.metric(
            "Expected Loss",
            f"₹{expected_loss:,.0f}"
        )


    with r4:

        st.metric(
            "Indicative Interest Rate",
            f"{indicative_rate:.1f}%"
        )


    st.divider()


    # --------------------------------------------------
    # RISK DETAILS
    # --------------------------------------------------

    st.subheader("📌 Risk Calculation")


    c1, c2, c3 = st.columns(3)


    with c1:

        st.write("**Probability of Default (PD)**")

        st.write(
            f"{pd_percentage:.2f}%"
        )


    with c2:

        st.write("**Loss Given Default (LGD)**")

        st.write(
            "60.00%"
        )


    with c3:

        st.write("**Exposure at Default (EAD)**")

        st.write(
            f"₹{EAD:,.0f}"
        )


    st.info(
        f"""
        **Expected Loss = PD × LGD × EAD**
        
        = {pd_percentage:.2f}% × 60% × ₹{EAD:,.0f}
        
        = **₹{expected_loss:,.0f}**
        """
    )


    # --------------------------------------------------
    # DECISION SUMMARY
    # --------------------------------------------------

    st.subheader("📝 Assessment Summary")

    st.write(
        f"""
        The model estimates a **{pd_percentage:.2f}% probability of default**
        for this application.
        
        The resulting risk classification is **{risk_band}**.
        
        Based on the project's illustrative risk-based pricing framework,
        the indicative interest rate is **{indicative_rate:.1f}%**.
        """
    )


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    """
    ⚠️ Disclaimer: This is an educational/portfolio project using synthetic
    data. The model and pricing assumptions are illustrative and should not
    be used for actual lending or credit decisions.
    """
)
