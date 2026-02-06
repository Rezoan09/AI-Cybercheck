import os
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_artifact
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from phishing_features import extract_phishing_features

st.set_page_config(
    page_title="AI CyberCheck Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: rgba(28, 131, 225, 0.1);
    }
    .threat-high {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .threat-low {
        background: linear-gradient(90deg, #00cc88 0%, #00ff9d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00ff9d;'>🛡️ AI CyberCheck Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Real-time Network Threat Detection & Analysis</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/security-shield-green.png", width=150)
    st.markdown("## 🔐 Security Center")
    st.markdown("### System Status")
    
    ids_ready = os.path.exists("models/ids_model.pkl") and os.path.exists("models/ids_columns.pkl")
    phish_ready = os.path.exists("models/phish_model.pkl")
    
    if ids_ready:
        st.success("✅ IDS Model: Active")
    else:
        st.error("❌ IDS Model: Offline")
    
    if phish_ready:
        st.success("✅ Phishing Detector: Active")
    else:
        st.error("❌ Phishing Detector: Offline")
    
    st.markdown("---")
    st.markdown("### 📊 Detection Stats")
    if 'total_scans' not in st.session_state:
        st.session_state.total_scans = 0
    if 'threats_detected' not in st.session_state:
        st.session_state.threats_detected = 0
    
    st.metric("Total Scans", st.session_state.total_scans)
    st.metric("Threats Detected", st.session_state.threats_detected, 
              delta=None if st.session_state.threats_detected == 0 else "⚠️")

tab1, tab2 = st.tabs(["🌐 Network Intrusion Detection", "📧 Phishing Detection"])

with tab1:
    if not ids_ready:
        st.error("⚠️ IDS Model not loaded. Please train the model first.")
    else:
        clf = load_artifact("ids_model", "models")
        cols = load_artifact("ids_columns", "models")
        
        st.markdown("## 🔍 Network Traffic Analysis")
        
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Single Flow Analysis", "Batch File Upload"],
            horizontal=True
        )
        
        if analysis_mode == "Single Flow Analysis":
            st.markdown("### Enter Network Flow Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Flow Timing**")
                flow_duration = st.number_input(
                    "Flow Duration (μs)",
                    min_value=0,
                    value=1000,
                    step=100,
                    help="Duration of network flow in microseconds"
                )
            
            with col2:
                st.markdown("**📤 Forward Traffic**")
                fwd_packets = st.number_input(
                    "Total Fwd Packets",
                    min_value=0,
                    value=10,
                    step=1,
                    help="Number of forward packets"
                )
                fwd_length_mean = st.number_input(
                    "Fwd Packet Length Mean",
                    min_value=0.0,
                    value=420.0,
                    step=10.0,
                    help="Average size of forward packets"
                )
            
            with col3:
                st.markdown("**📥 Backward Traffic**")
                bwd_packets = st.number_input(
                    "Total Backward Packets",
                    min_value=0,
                    value=8,
                    step=1,
                    help="Number of backward packets"
                )
                bwd_length_mean = st.number_input(
                    "Bwd Packet Length Mean",
                    min_value=0.0,
                    value=380.0,
                    step=10.0,
                    help="Average size of backward packets"
                )
            
            flow_bytes_s = st.number_input(
                "Flow Bytes/s",
                min_value=0.0,
                value=12000.0,
                step=100.0,
                help="Flow rate in bytes per second"
            )
            
            if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
                st.session_state.total_scans += 1
                
                input_data = {
                    cols[0]: flow_duration,
                    cols[1]: fwd_packets,
                    cols[2]: bwd_packets,
                    cols[3]: fwd_length_mean,
                    cols[4]: bwd_length_mean,
                    cols[5]: flow_bytes_s
                }
                
                df = pd.DataFrame([input_data])
                pred = clf.predict(df)[0]
                
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(df)[0]
                    confidence = float(proba[pred]) * 100
                else:
                    confidence = 50.0
                
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pred == 1:
                        st.session_state.threats_detected += 1
                        st.markdown('<div class="threat-high">🚨 THREAT DETECTED</div>', unsafe_allow_html=True)
                        threat_level = "HIGH RISK"
                        threat_color = "#ff4b4b"
                    else:
                        st.markdown('<div class="threat-low">✅ TRAFFIC NORMAL</div>', unsafe_allow_html=True)
                        threat_level = "LOW RISK"
                        threat_color = "#00cc88"
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        title = {'text': "Confidence Level"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': threat_color},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.markdown(f"### Threat Analysis")
                    st.metric("Classification", "ATTACK" if pred == 1 else "BENIGN")
                    st.metric("Risk Level", threat_level)
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("---")
                st.markdown("### 📊 Feature Analysis")
                
                feature_values = [flow_duration/1000, fwd_packets, bwd_packets, 
                                fwd_length_mean, bwd_length_mean, flow_bytes_s/1000]
                feature_names = ["Flow Duration\n(ms)", "Fwd Packets", "Bwd Packets",
                               "Fwd Pkt Size", "Bwd Pkt Size", "Flow Rate\n(KB/s)"]
                
                fig = go.Figure(data=[
                    go.Bar(x=feature_names, y=feature_values,
                          marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b'])
                ])
                fig.update_layout(
                    title="Network Flow Characteristics",
                    xaxis_title="Features",
                    yaxis_title="Values",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔬 Detailed Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Traffic Pattern:**
                    - Total Packets: {fwd_packets + bwd_packets}
                    - Packet Ratio (Fwd:Bwd): {fwd_packets}:{bwd_packets}
                    - Average Packet Size: {(fwd_length_mean + bwd_length_mean)/2:.2f} bytes
                    """)
                
                with col2:
                    st.info(f"""
                    **Flow Metrics:**
                    - Duration: {flow_duration/1000:.2f} ms
                    - Throughput: {flow_bytes_s/1024:.2f} KB/s
                    - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """)
                
                if pred == 1:
                    st.error("""
                    ⚠️ **Security Recommendation:**
                    This traffic pattern exhibits characteristics consistent with network attacks (DDoS or Port Scanning).
                    Recommended actions:
                    - Monitor this source IP for continued suspicious activity
                    - Consider implementing rate limiting
                    - Review firewall rules
                    - Alert security team for further investigation
                    """)
        
        else:  # Batch Upload
            st.markdown("### 📁 Batch Analysis from CSV File")
            st.info(f"Expected columns: {', '.join(cols)}")
            
            uploaded = st.file_uploader(
                "Upload Network Traffic CSV",
                type=["csv"],
                help="CSV file should contain the 6 required network flow features"
            )
            
            if uploaded:
                df = pd.read_csv(uploaded)
                st.success(f"✅ File loaded: {len(df)} network flows")
                
                st.markdown("#### Preview of uploaded data:")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🔍 Analyze All Flows", type="primary"):
                    for c in cols:
                        if c not in df.columns:
                            df[c] = 0
                    
                    df_model = df[cols].fillna(0)
                    preds = clf.predict(df_model)
                    
                    if hasattr(clf, "predict_proba"):
                        probas = clf.predict_proba(df_model)
                        confidences = [probas[i][preds[i]] * 100 for i in range(len(preds))]
                    else:
                        confidences = [50.0] * len(preds)
                    
                    results_df = df.copy()
                    results_df['Prediction'] = ['ATTACK' if p == 1 else 'BENIGN' for p in preds]
                    results_df['Threat_Level'] = ['HIGH' if p == 1 else 'LOW' for p in preds]
                    results_df['Confidence'] = [f"{c:.1f}%" for c in confidences]
                    
                    st.markdown("---")
                    st.markdown("## 📊 Batch Analysis Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(preds)
                    threats = sum(preds)
                    benign = total - threats
                    threat_rate = (threats/total)*100 if total > 0 else 0
                    
                    with col1:
                        st.metric("Total Flows", total)
                    with col2:
                        st.metric("Threats Detected", threats, delta="⚠️" if threats > 0 else "✅")
                    with col3:
                        st.metric("Benign Traffic", benign)
                    with col4:
                        st.metric("Threat Rate", f"{threat_rate:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Benign', 'Attack'],
                            values=[benign, threats],
                            hole=0.4,
                            marker_colors=['#00cc88', '#ff4b4b']
                        )])
                        fig.update_layout(
                            title="Traffic Distribution",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        confidence_dist = pd.DataFrame({
                            'Range': ['0-50%', '50-75%', '75-90%', '90-100%'],
                            'Count': [
                                sum(1 for c in confidences if c < 50),
                                sum(1 for c in confidences if 50 <= c < 75),
                                sum(1 for c in confidences if 75 <= c < 90),
                                sum(1 for c in confidences if c >= 90)
                            ]
                        })
                        fig = px.bar(confidence_dist, x='Range', y='Count',
                                    title="Confidence Distribution",
                                    color='Count',
                                    color_continuous_scale='Viridis')
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"threat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                    
                    st.session_state.total_scans += total
                    st.session_state.threats_detected += threats

with tab2:
    if not phish_ready:
        st.error("⚠️ Phishing Model not loaded. Please train the model first.")
    else:
        clf = load_artifact("phish_model", "models")
        feature_names = load_artifact("phish_features", "models")
        
        st.markdown("## 📧 Email Phishing Detection")
        
        text = st.text_area(
            "Enter email or message text to analyze:",
            height=200,
            placeholder="Paste suspicious email content here..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🔍 Analyze for Phishing", type="primary", use_container_width=True)
        
        if analyze_btn:
            if text.strip():
                features_dict = extract_phishing_features(text)
                features_df = pd.DataFrame([features_dict])
                features_df = features_df[feature_names]
                
                pred = int(clf.predict(features_df)[0])
                
                if hasattr(clf, "predict_proba"):
                    probas = clf.predict_proba(features_df)[0]
                    confidence = float(probas[pred]) * 100
                else:
                    confidence = 50.0
                
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if pred == 1:
                        st.markdown('<div class="threat-high">🎣 PHISHING DETECTED</div>', unsafe_allow_html=True)
                        st.error("""
                        ⚠️ **Warning: This message appears to be a phishing attempt!**
                        
                        **Recommended Actions:**
                        - Do NOT click any links
                        - Do NOT provide personal information
                        - Do NOT download attachments
                        - Report to your security team
                        - Delete the message
                        """)
                    else:
                        st.markdown('<div class="threat-low">✅ MESSAGE APPEARS SAFE</div>', unsafe_allow_html=True)
                        st.success("""
                        ✅ **This message appears legitimate**
                        
                        However, always remain cautious and:
                        - Verify sender identity
                        - Check for suspicious links
                        - Be wary of urgent requests
                        """)
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        title = {'text': "Detection Confidence"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#ff4b4b" if pred == 1 else "#00cc88"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "darkgray"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📊 Analysis Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Classification", "PHISHING" if pred == 1 else "LEGITIMATE")
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                
                with col2:
                    st.metric("Message Length", f"{len(text)} characters")
                    st.metric("Word Count", f"{len(text.split())} words")
                
                st.markdown("### 🔬 Extracted Features")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Text Analysis:**
                    - Total Words: {features_dict['num_words']}
                    - Unique Words: {features_dict['num_unique_words']}
                    - Stopwords: {features_dict['num_stopwords']}
                    - Spelling Errors: {features_dict['num_spelling_errors']}
                    """)
                
                with col2:
                    st.info(f"""
                    **Suspicious Indicators:**
                    - Links Found: {features_dict['num_links']}
                    - Unique Domains: {features_dict['num_unique_domains']}
                    - Email Addresses: {features_dict['num_email_addresses']}
                    - Urgent Keywords: {features_dict['num_urgent_keywords']}
                    """)
                
            else:
                st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>AI CyberCheck | Developed by Rezoan Sultan</p>", unsafe_allow_html=True)
