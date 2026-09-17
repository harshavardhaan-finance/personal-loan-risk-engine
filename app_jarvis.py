import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="JARVIS | Personal Loan Risk Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(0,180,255,.10), transparent 28%),
                radial-gradient(circle at 85% 20%, rgba(120,80,255,.10), transparent 30%),
                #070b14;
    color: #e8f1ff;
}
.block-container {max-width:1250px; padding-top:2rem;}
.jarvis-header {
    border:1px solid rgba(80,190,255,.35); border-radius:18px;
    padding:24px 28px; background:linear-gradient(135deg,rgba(12,25,45,.95),rgba(10,14,28,.95));
    box-shadow:0 0 30px rgba(0,170,255,.08); margin-bottom:22px;
}
.jarvis-title {font-size:34px;font-weight:800;}
.jarvis-subtitle {color:#8eb8d8;margin-top:6px;}
.status {display:inline-block;margin-top:12px;padding:6px 12px;border-radius:999px;
    border:1px solid rgba(0,255,180,.35);color:#71ffd0;background:rgba(0,255,180,.06);
    font-size:12px;letter-spacing:1px;}
.panel {border:1px solid rgba(100,160,210,.22);border-radius:16px;padding:18px 20px;
    background:rgba(11,18,32,.78);margin-bottom:18px;}
.risk-card {border:1px solid rgba(100,180,255,.28);border-radius:15px;padding:18px;
    background:linear-gradient(145deg,rgba(17,29,49,.9),rgba(9,14,25,.9));text-align:center;min-height:120px;}
.risk-label {color:#8fa9c4;font-size:13px;text-transform:uppercase;letter-spacing:1px;}
.risk-value {font-size:28px;font-weight:800;margin-top:8px;color:#f1f7ff;}
.assistant {border-left:3px solid #38bdf8;border-radius:10px;padding:15px 18px;
    background:rgba(20,65,100,.18);margin-top:12px;}
div[data-testid="stButton"] > button {
    border-radius:12px;border:1px solid rgba(80,190,255,.45);
    background:linear-gradient(90deg,#0b6fa4,#1556a6);color:white;font-weight:700;min-height:48px;
}
[data-testid="stMetric"] {background:rgba(15,25,42,.72);border:1px solid rgba(100,160,210,.20);
    padding:12px;border-radius:14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("personal_loan_pd_model.pkl")

model = load_model()

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<div class="jarvis-header">
<div class="jarvis-title">◈ JARVIS — Personal Loan Risk Engine</div>
<div class="jarvis-subtitle">Intelligent credit-risk assessment • PD • Expected Loss • Risk-based Pricing</div>
<div class="status">● SYSTEM ONLINE &nbsp; | &nbsp; PD MODEL READY</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◈ JARVIS")
    st.caption("Credit Risk Intelligence Console")
    st.divider()
    st.write("🟢 Model: Online")
    st.write("🟢 Risk Engine: Online")
    st.write("🟢 Pricing Engine: Online")
    st.divider()
    st.markdown("### Model Architecture")
    st.caption("Logistic Regression • 7 borrower variables")
    st.caption("LGD assumption: 60%")
    st.caption("EAD: Requested loan amount")
    st.divider()
    st.caption("Educational portfolio application using synthetic data. Outputs are illustrative.")

st.markdown('<div class="panel"><h3>◈ Applicant Command Center</h3>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown("**Financial Profile**")
    annual_income = st.number_input("Annual Income (₹)", min_value=0.0, value=800000.0, step=25000.0, format="%.0f")
    existing_emi = st.number_input("Existing Monthly EMI (₹)", min_value=0.0, value=15000.0, step=1000.0, format="%.0f")
    net_worth = st.number_input("Net Worth (₹)", min_value=-50000000.0, value=1000000.0, step=25000.0, format="%.0f")
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=50000.0, max_value=2000000.0, value=500000.0, step=25000.0, format="%.0f")

with right:
    st.markdown("**Credit Profile**")
    credit_score = st.slider("Credit Score", 520, 850, 750, 1)
    credit_utilization = st.slider("Credit Utilization", 0.0, 100.0, 25.0, 1.0)
    dpd_30_count = st.number_input("30+ DPD Count", 0, 5, 0, 1)
    monthly_income = annual_income / 12
    foir = existing_emi / monthly_income * 100 if monthly_income > 0 else 0
    st.metric("Current FOIR", f"{foir:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

if st.button("◈  RUN JARVIS CREDIT ASSESSMENT", use_container_width=True):
    applicant = pd.DataFrame({
        "annual_income": [annual_income],
        "credit_score": [credit_score],
        "existing_emi": [existing_emi],
        "net_worth": [net_worth],
        "credit_utilization": [credit_utilization / 100],
        "dpd_30_count": [dpd_30_count],
        "loan_amount": [loan_amount]
    })

    pd_probability = float(model.predict_proba(applicant)[0][1])
    lgd = 0.60
    ead = loan_amount
    expected_loss = pd_probability * lgd * ead

    if pd_probability < 0.02:
        risk_band, rate, message = "LOW", 11.0, "Risk indicators are relatively contained under the V0 model."
    elif pd_probability < 0.05:
        risk_band, rate, message = "MODERATE", 13.0, "The model identifies a moderate level of estimated default risk."
    elif pd_probability < 0.10:
        risk_band, rate, message = "HIGH", 16.0, "The model identifies elevated estimated default risk."
    else:
        risk_band, rate, message = "VERY HIGH", 20.0, "The model identifies materially elevated estimated default risk."

    st.session_state.result = {
        "pd": pd_probability * 100,
        "band": risk_band,
        "rate": rate,
        "el": expected_loss,
        "lgd": lgd,
        "ead": ead,
        "message": message
    }

if st.session_state.result:
    r = st.session_state.result
    st.markdown("---")
    st.markdown("## ◈ JARVIS Assessment Output")
    st.markdown(f'<div class="assistant"><b>JARVIS:</b> Assessment complete. {r["message"]}</div>', unsafe_allow_html=True)
    st.write("")

    c1,c2,c3,c4 = st.columns(4)
    values = [
        ("Probability of Default", f'{r["pd"]:.2f}%'),
        ("Risk Classification", r["band"]),
        ("Expected Loss", f'₹{r["el"]:,.0f}'),
        ("Indicative Rate", f'{r["rate"]:.1f}%')
    ]
    for col, (label, value) in zip((c1,c2,c3,c4), values):
        with col:
            st.markdown(f'<div class="risk-card"><div class="risk-label">{label}</div><div class="risk-value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### ◈ Default Risk Monitor")
    st.progress(min(r["pd"] / 20, 1.0))
    st.caption(f'Estimated PD: {r["pd"]:.2f}% • Visual scale capped at 20%')

    tab1, tab2, tab3 = st.tabs(["◈ Risk Economics","◈ Applicant Diagnostics","◈ JARVIS Explanation"])

    with tab1:
        a,b,c = st.columns(3)
        a.metric("PD", f'{r["pd"]:.2f}%')
        b.metric("LGD", f'{r["lgd"]:.0%}')
        c.metric("EAD", f'₹{r["ead"]:,.0f}')
        st.info(f'Expected Loss = PD × LGD × EAD = {r["pd"]:.2f}% × 60% × ₹{r["ead"]:,.0f} = ₹{r["el"]:,.0f}')

    with tab2:
        d1,d2 = st.columns(2)
        with d1:
            st.write(f"Annual income: ₹{annual_income:,.0f}")
            st.write(f"Credit score: {credit_score}")
            st.write(f"Existing EMI: ₹{existing_emi:,.0f}")
            st.write(f"Net worth: ₹{net_worth:,.0f}")
        with d2:
            st.write(f"Credit utilization: {credit_utilization:.0f}%")
            st.write(f"30+ DPD count: {dpd_30_count}")
            st.write(f"FOIR: {foir:.1f}%")
            st.write(f"Requested loan: ₹{loan_amount:,.0f}")

    with tab3:
        st.markdown(f'''<div class="assistant"><b>JARVIS:</b><br><br>
        The estimated Probability of Default is <b>{r["pd"]:.2f}%</b>.<br><br>
        Using an illustrative LGD of <b>60%</b> and EAD of <b>₹{r["ead"]:,.0f}</b>,
        Expected Loss is <b>₹{r["el"]:,.0f}</b>.<br><br>
        The V0 pricing framework maps this risk to an illustrative rate of <b>{r["rate"]:.1f}%</b>.
        </div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ Synthetic-data educational portfolio project. Outputs are illustrative and not actual lending or credit decisions.")
