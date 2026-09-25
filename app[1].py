import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier, IsolationForest

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    margin-bottom: 20px;
}

.hero h1 {
    margin-bottom: 5px;
}

.card {
    padding: 18px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}

.result-high {
    padding: 18px;
    border-radius: 15px;
    background: #fee2e2;
    border: 1px solid #fecaca;
}

.result-medium {
    padding: 18px;
    border-radius: 15px;
    background: #fef3c7;
    border: 1px solid #fde68a;
}

.result-low {
    padding: 18px;
    border-radius: 15px;
    background: #dcfce7;
    border: 1px solid #bbf7d0;
}

.small-text {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA + MODEL
# =========================================================

@st.cache_resource
def train_models():

    data = {
        "Amount": [
            100, 200, 150, 5000, 300,
            10000, 250, 7000, 120, 8000,
            400, 15000, 180, 9000, 350,
            20000, 220, 6000, 450, 12000
        ],

        "Transactions_1H": [
            1, 2, 1, 10, 2,
            15, 1, 12, 1, 14,
            2, 18, 1, 13, 2,
            20, 2, 11, 3, 16
        ],

        "Previous_Fraud": [
            0, 0, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1
        ],

        "Distance_KM": [
            2, 5, 3, 500, 4,
            800, 6, 600, 3, 700,
            5, 900, 2, 650, 4,
            1000, 5, 550, 3, 750
        ],

        "Fraud": [
            0, 0, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1
        ]
    }

    df = pd.DataFrame(data)

    features = [
        "Amount",
        "Transactions_1H",
        "Previous_Fraud",
        "Distance_KM"
    ]

    X = df[features]
    y = df["Fraud"]

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=21
    )

    rf_model.fit(X, y)

    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.20,
        random_state=21
    )

    anomaly_detector.fit(X)

    return df, features, rf_model, anomaly_detector


df, features, rf_model, anomaly_detector = train_models()

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🛡️ Real-Time Fraud Detection System</h1>
    <div>AI-powered transaction risk analysis using Random Forest + Isolation Forest</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ System Info")
    st.write("**Fraud Model:** Random Forest")
    st.write("**Anomaly Model:** Isolation Forest")
    st.write("**Features:** 4")
    st.write("**Decision Mode:** Real-Time")

    st.divider()

    st.info(
        "Enter a transaction and click Analyze Transaction "
        "to get fraud probability, risk level, anomaly status "
        "and an explanation."
    )

# =========================================================
# TOP METRICS
# =========================================================

total = len(df)
frauds = int(df["Fraud"].sum())
normal = total - frauds

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", total)
c2.metric("Fraud Transactions", frauds)
c3.metric("Normal Transactions", normal)
c4.metric("Fraud Rate", f"{frauds / total * 100:.1f}%")

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("💳 Transaction Details")

left, right = st.columns(2)

with left:
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=9000.0,
        step=100.0
    )

    transactions_1h = st.number_input(
        "Transactions in Last 1 Hour",
        min_value=0,
        value=14,
        step=1
    )

with right:
    previous_fraud = st.selectbox(
        "Previous Fraud History",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    distance = st.number_input(
        "Distance (KM)",
        min_value=0.0,
        value=700.0,
        step=10.0
    )

analyze = st.button(
    "🔍 ANALYZE TRANSACTION",
    use_container_width=True,
    type="primary"
)

# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    new_transaction = pd.DataFrame([[
        amount,
        transactions_1h,
        previous_fraud,
        distance
    ]], columns=features)

    start_time = time.perf_counter()

    prediction = int(rf_model.predict(new_transaction)[0])

    probability = float(
        rf_model.predict_proba(new_transaction)[0][1] * 100
    )

    anomaly_result = int(
        anomaly_detector.predict(new_transaction)[0]
    )

    latency = (time.perf_counter() - start_time) * 1000

    # Risk
    if probability >= 75:
        risk = "HIGH"
        risk_class = "result-high"
    elif probability >= 40:
        risk = "MEDIUM"
        risk_class = "result-medium"
    else:
        risk = "LOW"
        risk_class = "result-low"

    # Decision
    if prediction == 1:
        decision = "BLOCK / REVIEW"
    else:
        decision = "ALLOW"

    # Explanation
    reasons = []

    if amount > 5000:
        reasons.append("High transaction amount")

    if transactions_1h > 8:
        reasons.append("High transaction frequency")

    if previous_fraud == 1:
        reasons.append("Previous fraud history")

    if distance > 300:
        reasons.append("Unusual geographic distance")

    if len(reasons) == 0:
        explanation = "No major suspicious signal detected."
    else:
        explanation = ", ".join(reasons)

    anomaly_status = (
        "ANOMALOUS" if anomaly_result == -1 else "NORMAL"
    )

    # Save history
    st.session_state.history.insert(0, {
        "Amount": amount,
        "Transactions_1H": transactions_1h,
        "Previous_Fraud": previous_fraud,
        "Distance_KM": distance,
        "Fraud Probability": round(probability, 2),
        "Risk": risk,
        "Decision": decision,
        "Anomaly": anomaly_status,
        "Latency (ms)": round(latency, 4)
    })

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("📊 Analysis Result")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Fraud Probability",
        f"{probability:.2f}%"
    )

    r2.metric(
        "Risk Level",
        risk
    )

    r3.metric(
        "Anomaly",
        anomaly_status
    )

    r4.metric(
        "Latency",
        f"{latency:.3f} ms"
    )

    st.markdown(
        f"""
        <div class="{risk_class}">
            <h3>Decision: {decision}</h3>
            <p><b>Explanation:</b> {explanation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # RISK BAR
    # =====================================================

    st.write("### Risk Score")

    st.progress(
        min(int(probability), 100)
    )

    # =====================================================
    # SIGNALS
    # =====================================================

    st.write("### 🔎 Transaction Signals")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Amount", f"₹{amount:,.0f}")
    s2.metric("Transactions / 1H", int(transactions_1h))
    s3.metric(
        "Previous Fraud",
        "Yes" if previous_fraud else "No"
    )
    s4.metric("Distance", f"{distance:,.0f} KM")

# =========================================================
# RECENT TRANSACTIONS
# =========================================================

st.divider()
st.subheader("🧾 Recent Analysis History")

if len(st.session_state.history) == 0:
    st.info("No transactions analyzed yet.")
else:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.divider()
st.subheader("🧠 Model Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance.set_index("Feature")
)

# =========================================================
# DATASET PREVIEW
# =========================================================

with st.expander("📁 View Training Dataset"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "PS-02 Demo • Real-Time Fraud & Anomaly Detection • "
    "Academic / demonstration implementation"
)
