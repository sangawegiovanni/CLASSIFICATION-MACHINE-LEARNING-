from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib, numpy as np

app = FastAPI()
model = joblib.load("model.pkl")
enc = joblib.load("encoder.pkl")

def e(v):
    return enc.transform([v])[0] if v in enc.classes_ else -1

HTML = """
<form method='post' action='/predict'>
  Step: <input name='step'><br>
  Type: <select name='type'><option>PAYMENT</option><option>TRANSFER</option><option>CASH_OUT</option><option>CASH_IN</option><option>DEBIT</option></select><br>
  Amount: <input name='amount'><br>
  nameOrig: <input name='nameOrig'><br>
  oldBalOrig: <input name='oldbalanceOrig'><br>
  newBalOrig: <input name='newbalanceOrig'><br>
  nameDest: <input name='nameDest'><br>
  oldBalDest: <input name='oldbalanceDest'><br>
  newBalDest: <input name='newbalanceDest'><br>
  <button type='submit'>Predict</button>
</form>
{result}
"""

@app.get("/", response_class=HTMLResponse)
def home(): return HTML.format(result="")

@app.post("/predict", response_class=HTMLResponse)
def predict(step:int=Form(...), type:str=Form(...), amount:float=Form(...),
            nameOrig:str=Form(...), oldbalanceOrig:float=Form(...), newbalanceOrig:float=Form(...),
            nameDest:str=Form(...), oldbalanceDest:float=Form(...), newbalanceDest:float=Form(...)):
    X = np.array([[step, e(type), amount, e(nameOrig), oldbalanceOrig, newbalanceOrig, e(nameDest), oldbalanceDest, newbalanceDest]])
    r = "🚨 FRAUD" if model.predict(X)[0] == 1 else "✅ LEGIT"
    return HTML.format(result=f"<h2>{r}</h2>")