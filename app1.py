from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib, numpy as np

app = FastAPI()
model = joblib.load("model.pkl")
enc = joblib.load("encoder.pkl")

def e(v):
    return enc.transform([v])[0] if v in enc.classes_ else -1

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detector</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }}

  .card {{
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    width: 100%;
    max-width: 480px;
  }}

  h1 {{
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.05em;
    margin-bottom: 1.75rem;
    text-transform: uppercase;
  }}

  .field {{ margin-bottom: 1rem; }}

  label {{
    display: block;
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  input, select {{
    width: 100%;
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 6px;
    color: #e2e8f0;
    padding: 0.55rem 0.75rem;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.15s;
  }}

  input:focus, select:focus {{ border-color: #4f6ef7; }}
  select option {{ background: #1a1d27; }}

  button {{
    width: 100%;
    margin-top: 1.5rem;
    padding: 0.7rem;
    background: #4f6ef7;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }}

  button:hover {{ background: #3b5bdb; }}

  .result {{
    margin-top: 1.5rem;
    padding: 0.9rem 1rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
  }}

  .fraud {{ background: #3b1219; color: #f87171; border: 1px solid #7f1d1d; }}
  .legit {{ background: #0d2b1f; color: #4ade80; border: 1px solid #14532d; }}
</style>
</head>
<body>
<div class="card">
  <h1>Transaction Check</h1>
  <form method="post" action="/predict">
    <div class="field"><label>Step</label><input name="step" type="number" required></div>
    <div class="field">
      <label>Type</label>
      <select name="type">
        <option>PAYMENT</option><option>TRANSFER</option>
        <option>CASH_OUT</option><option>CASH_IN</option><option>DEBIT</option>
      </select>
    </div>
    <div class="field"><label>Amount</label><input name="amount" type="number" step="any" required></div>
    <div class="field"><label>Sender Old Balance</label><input name="oldbalanceOrig" type="number" step="any" required></div>
    <div class="field"><label>Sender New Balance</label><input name="newbalanceOrig" type="number" step="any" required></div>
    <div class="field"><label>Receiver Old Balance</label><input name="oldbalanceDest" type="number" step="any" required></div>
    <div class="field"><label>Receiver New Balance</label><input name="newbalanceDest" type="number" step="any" required></div>
    <button type="submit">Run Detection</button>
  </form>
  {result}
</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML.format(result="")

@app.post("/predict", response_class=HTMLResponse)
def predict(
    step: int = Form(...), type: str = Form(...), amount: float = Form(...),
    oldbalanceOrig: float = Form(...), newbalanceOrig: float = Form(...),
    oldbalanceDest: float = Form(...), newbalanceDest: float = Form(...)
):
    X = np.array([[step, e(type), amount, -1, oldbalanceOrig, newbalanceOrig,
                   -1, oldbalanceDest, newbalanceDest]])
    pred = model.predict(X)[0]
    label = "fraud" if pred == 1 else "legit"
    text  = "🚨 FRAUD DETECTED" if pred == 1 else "✅ LEGITIMATE"
    return HTML.format(result=f'<div class="result {label}">{text}</div>')