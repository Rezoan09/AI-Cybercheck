
# AI CyberCheck (Advanced Build — Dataset Ready)

**Components**
- `model_train.py` — trains (1) IDS RandomForest on CSV; (2) Phishing TF-IDF + Logistic Regression
- `main.py` — Flask API (`/detect_anomaly`, `/detect_phishing`)
- `dashboard.py` — Streamlit UI for testing/presenting
- `utils.py` — helpers
- `models/` — created after training, stores pickles

## Replit Setup
```bash
pip install -r requirements.txt

# Train with your datasets (or run without args to use built-in demos)
python model_train.py --ids_csv network_sample.csv --ids_label Label \
  --drop_cols "Flow_ID, Timestamp, Src_IP, Src_Port, Dst_IP, Dst_Port, Protocol"

python model_train.py --phish_csv phish_sample.csv --phish_text_col message --phish_label_col label
```

Run API:
```bash
python main.py
```

Run dashboard:
```bash
streamlit run dashboard.py --server.port=3000 --server.address=0.0.0.0
```

## API Examples
### /detect_anomaly
```json
{
  "rows": [
    {"Flow_Duration": 1000, "Total_Fwd_Packets": 10, "Total_Bwd_Packets": 8, "Fwd_Packet_Length_Mean": 420, "Bwd_Packet_Length_Mean": 380, "Flow_Bytes_s": 12000}
  ]
}
```
### /detect_phishing
```json
{
  "text": "URGENT: Your password will expire today. Click here to verify."
}
```
