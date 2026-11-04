import io, json, math, sqlite3, hashlib, secrets
import numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="DOOMSTERS Retail Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- auth (SQLite, salted PBKDF2) ----------------
db = sqlite3.connect("doomsters.db", check_same_thread=False)
db.execute("create table if not exists users(email text primary key, name text, pw text, salt text)")
SESS = {}
def hp(p, s): return hashlib.pbkdf2_hmac("sha256", p.encode(), s.encode(), 100_000).hex()
class Reg(BaseModel):
    email: str
    password: str
    name: str = ""
class Login(BaseModel):
    email: str
    password: str
def issue(email, name):
    t = secrets.token_hex(16); SESS[t] = email
    return {"token": t, "user": {"email": email, "name": name}}
@app.post("/api/auth/register")
def register(r: Reg):
    if "@" not in r.email or len(r.password) < 6:
        raise HTTPException(400, "Enter a valid email and a password of at least 6 characters.")
    if db.execute("select 1 from users where email=?", (r.email,)).fetchone():
        raise HTTPException(409, "That email is already registered. Sign in instead.")
    name = r.name or r.email.split("@")[0]; s = secrets.token_hex(8)
    db.execute("insert into users values(?,?,?,?)", (r.email, name, hp(r.password, s), s)); db.commit()
    return issue(r.email, name)
@app.post("/api/auth/login")
def login(r: Login):
    row = db.execute("select name,pw,salt from users where email=?", (r.email,)).fetchone()
    if not row or hp(r.password, row[2]) != row[1]:
        raise HTTPException(401, "Email or password is incorrect.")
    return issue(r.email, row[0])
@app.post("/api/auth/demo")
def demo_login(): return issue("demo@doomsters.app", "Demo Analyst")

# ---------------- data ----------------
DEMO = "Synthetic demo dataset (generated, not real business data)"
REQ = ["date", "product", "category", "state", "city", "units", "price"]
PRODUCTS = {"Aero Earbuds": ("Electronics", 7200), "Pulse Watch": ("Electronics", 11900), "Volt Charger": ("Electronics", 1900),
            "Nimbus Hoodie": ("Apparel", 4300), "Trail Jacket": ("Apparel", 9500), "Everyday Tee": ("Apparel", 1400),
            "Brew Press": ("Home", 3100), "Oak Lamp": ("Home", 5100), "Linen Throw": ("Home", 3300),
            "Glow Serum": ("Beauty", 2900), "Calm Balm": ("Beauty", 1100), "Silk Mask": ("Beauty", 1700)}
CITIES = {"Mumbai": "Maharashtra", "Pune": "Maharashtra", "Bengaluru": "Karnataka", "Delhi": "Delhi",
          "Kolkata": "West Bengal", "Chennai": "Tamil Nadu", "Hyderabad": "Telangana", "Ahmedabad": "Gujarat"}
S = {"df": None, "src": DEMO, "raw": None}
CACHE = {}

def prepare(df):
    df = df.copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("units", "price"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "product", "units", "price"])
    df["date"] = df["date"].dt.normalize()
    if "order_id" not in df: df["order_id"] = np.arange(1, len(df) + 1)
    if "store" not in df: df["store"] = df["city"].astype(str) + " Store"
    if "stock" in df: df["stock"] = pd.to_numeric(df["stock"], errors="coerce")
    df["revenue"] = df.units * df.price
    return df.sort_values("date").reset_index(drop=True)

def make_demo():
    rng = np.random.default_rng(7); days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=540)
    cities = list(CITIES); rows = []; oid = 0
    stock = {}
    for pi, (p, (c, pr)) in enumerate(PRODUCTS.items()):
        base = rng.uniform(5, 16); stock[p] = int(base * rng.choice([2, 4, 9, 25, 60, 140]))
        for i, d in enumerate(days):
            lam = base * (1 + 0.0006 * i) * (1.35 if d.dayofweek >= 5 else 1) * (1 + 0.25 * np.sin(2 * np.pi * d.dayofyear / 365 + pi)) * (1.5 if d.month in (10, 11, 12) else 1)
            for city in rng.choice(cities, 2, replace=False):
                u = rng.poisson(lam / 2)
                if u:
                    oid += 1
                    rows.append((d, oid, int(rng.pareto(1.4) * 90) % 1500 + 1, p, c, city + " Store", CITIES[city], city, u, pr, stock[p]))
    return prepare(pd.DataFrame(rows, columns=["date", "order_id", "customer_id", "product", "category", "store", "state", "city", "units", "price", "stock"]))

S["df"] = make_demo()

def D(category=None, state=None):
    df = S["df"]
    if category and category != "All": df = df[df.category == category]
    if state and state != "All": df = df[df.state == state]
    if df.empty: raise HTTPException(404, "No data matches these filters.")
    return df

def rec(df): return json.loads(df.to_json(orient="records", date_format="iso"))
def dstr(df):
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"): df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df
def growth(df, col="revenue"):
    end = df.date.max(); cur = df[df.date > end - pd.Timedelta(days=30)][col].sum()
    prev = df[(df.date <= end - pd.Timedelta(days=30)) & (df.date > end - pd.Timedelta(days=60))][col].sum()
    return round(float((cur - prev) / prev * 100), 1) if prev else None

@app.get("/api/meta")
def meta():
    df = S["df"]
    return {"source": S["src"], "is_demo": S["src"] == DEMO, "rows": len(df), "from": str(df.date.min().date()), "to": str(df.date.max().date()),
            "categories": sorted(df.category.unique().tolist()), "states": sorted(df.state.unique().tolist()),
            "products": sorted(df["product"].unique().tolist()), "stores": sorted(df.store.unique().tolist()),
            "has_customers": "customer_id" in df, "has_stock": "stock" in df}

# ---------------- forecasting (Random Forest + Moving Average baseline) ----------------
def _x(y, i, d): return [d.dayofweek, d.month, i, y[i - 7], y[i - 14], np.mean(y[i - 7:i]), np.mean(y[i - 28:i])]
def _fit(y, idx):
    X = [_x(y, i, idx[i]) for i in range(28, len(y))]
    return RandomForestRegressor(120, min_samples_leaf=2, random_state=0, n_jobs=1).fit(X, y[28:])
def _roll(m, y, last, h):
    y = list(y)
    for k in range(h):
        y.append(max(0.0, float(m.predict([_x(y, len(y), last + pd.Timedelta(days=k + 1))])[0])))
    return np.array(y[-h:])

def run_forecast(df, horizon):
    s = df.groupby("date").units.sum().asfreq("D", fill_value=0); y = s.values.astype(float); H = 28
    if len(y) < 90: raise HTTPException(422, "At least 90 days of history are needed to train and backtest the model.")
    tr = y[:-H]; bt = _roll(_fit(tr, s.index[:-H]), tr, s.index[-H - 1], H); ma = np.full(H, tr[-7:].mean()); act = y[-H:]
    def met(p):
        e = act - p; nz = act > 0
        return {"mae": round(float(abs(e).mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2), "mape": round(float((abs(e[nz]) / act[nz]).mean() * 100), 2)}
    fc = _roll(_fit(y, s.index), y, s.index[-1], horizon); sd = float(np.std(act - bt))
    fd = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
    series = [{"date": str(d.date()), "actual": float(v)} for d, v in list(zip(s.index, y))[-56:-H]]
    series += [{"date": str(d.date()), "actual": float(a), "rf": round(float(r), 1), "ma": round(float(m), 1)} for d, a, r, m in zip(s.index[-H:], act, bt, ma)]
    for k, (d, v) in enumerate(zip(fd, fc)):
        w = 1.96 * sd * math.sqrt(1 + k / 30)
        series.append({"date": str(d.date()), "forecast": round(float(v), 1), "band": [round(max(0, v - w), 1), round(v + w, 1)]})
    price = float(df.revenue.sum() / max(df.units.sum(), 1)); recent = float(y[-28:].mean())
    return {"model": "Random Forest Regression (scikit-learn), trained on this dataset", "kind": "trained", "horizon": horizon,
            "backtest_days": H, "metrics": {"random_forest": met(bt), "moving_average_7d": met(ma)}, "series": series,
            "total_units": round(float(fc.sum())), "total_revenue": round(float(fc.sum()) * price),
            "trend_pct": round((float(fc.mean()) - recent) / recent * 100, 1) if recent else None,
            "interval_note": "Approximate 95% interval from the spread of backtest errors."}

@app.get("/api/forecast")
def forecast(horizon: int = 30, by: str = "all", value: str = "", category: str = None, state: str = None):
    if horizon not in (7, 30, 90): raise HTTPException(400, "Horizon must be 7, 30 or 90 days.")
    key = (horizon, by, value, category, state, S["src"], len(S["df"]))
    if key not in CACHE:
        df = D(category, state)
        if by in ("product", "category", "store") and value:
            df = df[df[by] == value]
            if df.empty: raise HTTPException(404, "No data for that selection.")
        CACHE[key] = run_forecast(df, horizon)
    return CACHE[key]

# ---------------- inventory ----------------
def inv_rows(df):
    if "stock" not in df or df.stock.isna().all(): return None
    end = df.date.max(); r = df[df.date > end - pd.Timedelta(days=28)]; out = []
    for p, g in df.groupby("product"):
        st = g.sort_values("date").stock.iloc[-1]
        if pd.isna(st): continue
        dd = r[r["product"] == p].units.sum() / 28; cover = st / dd if dd > 0 else None
        yr = g[g.date > end - pd.Timedelta(days=365)].units.sum()
        status = "Out of stock" if st <= 0 else "Low stock" if cover is not None and cover < 7 else "Overstocked" if cover is not None and cover > 120 else "Healthy"
        out.append({"product": p, "category": g.category.iloc[0], "stock": int(st), "daily_demand": round(float(dd), 1),
                    "days_cover": round(float(cover), 1) if cover is not None else None, "turnover": round(float(yr / st), 1) if st > 0 else None,
                    "status": status, "reorder_qty": max(0, math.ceil(dd * 37 - st))})
    return out

@app.get("/api/inventory")
def inventory(category: str = None, state: str = None):
    rows = inv_rows(D(category, state))
    if rows is None: raise HTTPException(422, "This dataset has no 'stock' column, so inventory analysis is unavailable.")
    return {"basis": "Trailing 28-day demand; reorder covers a 7-day lead time plus 30 days of demand.", "rows": rows}

# ---------------- overview / sales / customers / geo ----------------
@app.get("/api/overview")
def overview(category: str = None, state: str = None):
    df = D(category, state); rev = float(df.revenue.sum()); orders = int(df.order_id.nunique())
    m = df.groupby(df.date.dt.to_period("M")).revenue.sum().tail(13)
    try: fc = forecast(30, "all", "", category, state); f_rev, acc = fc["total_revenue"], round(100 - fc["metrics"]["random_forest"]["mape"], 1)
    except HTTPException: f_rev = acc = None
    rows = inv_rows(df); health = round(sum(r["status"] == "Healthy" for r in rows) / len(rows) * 100) if rows else None
    return {"revenue": rev, "units": int(df.units.sum()), "orders": orders, "aov": rev / max(orders, 1), "forecast_revenue": f_rev,
            "growth": growth(df), "inventory_health": health, "forecast_accuracy": acc,
            "monthly": [{"month": str(k), "revenue": float(v)} for k, v in m.items()],
            "recent": rec(dstr(df.tail(8)[["date", "order_id", "product", "city", "units", "revenue"]].iloc[::-1]))}

@app.get("/api/sales")
def sales(freq: str = "D", category: str = None, state: str = None):
    if freq not in ("D", "W", "M"): raise HTTPException(400, "freq must be D, W or M.")
    df = D(category, state); rule = {"D": "D", "W": "W", "M": "MS"}[freq]
    s = df.set_index("date")[["revenue", "units"]].resample(rule).sum().tail({"D": 90, "W": 52, "M": 24}[freq]).reset_index()
    s["date"] = s["date"].dt.strftime("%Y-%m-%d")
    def by(col): return rec(df.groupby(col).agg(revenue=("revenue", "sum"), units=("units", "sum")).sort_values("revenue", ascending=False).reset_index())
    prods = df.groupby("product").agg(category=("category", "first"), revenue=("revenue", "sum"), units=("units", "sum")).reset_index()
    prods["share"] = (prods.revenue / prods.revenue.sum() * 100).round(1)
    prods["growth"] = [growth(df[df["product"] == p], "units") for p in prods["product"]]
    wd = df.groupby(df.date.dt.dayofweek).revenue.sum().reindex(range(7), fill_value=0)
    mo = df.groupby(df.date.dt.month_name()).revenue.sum().sort_values(ascending=False)
    return {"series": rec(s), "products": rec(prods.sort_values("revenue", ascending=False)), "categories": by("category"),
            "weekday": [{"day": d, "revenue": float(v)} for d, v in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], wd)],
            "peak_month": mo.index[0], "peak_day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(wd.values.argmax())]}

@app.get("/api/customers")
def customers(category: str = None, state: str = None):
    df = D(category, state)
    if "customer_id" not in df: raise HTTPException(422, "This dataset has no 'customer_id' column, so customer analysis is unavailable.")
    end = df.date.max()
    g = df.groupby("customer_id").agg(orders=("order_id", "nunique"), spend=("revenue", "sum"), last=("date", "max"), first=("date", "min"))
    rec_d = (end - g["last"]).dt.days
    g["segment"] = np.select([rec_d > 120, (g.spend >= g.spend.quantile(.8)) & (rec_d <= 60), g.orders == 1], ["Lapsed", "Champions", "New / one-time"], "Regular")
    mf = g["first"].dt.to_period("M"); act = df.assign(m=df.date.dt.to_period("M")).groupby("m").customer_id.nunique()
    new = mf.value_counts().reindex(act.index, fill_value=0)
    cur = set(df[df.date > end - pd.Timedelta(days=90)].customer_id); prev = set(df[(df.date <= end - pd.Timedelta(days=90)) & (df.date > end - pd.Timedelta(days=180))].customer_id)
    return {"customers": len(g), "avg_spend": float(g.spend.mean()), "orders_per_customer": round(float(g.orders.mean()), 2),
            "retention_90d": round(len(cur & prev) / len(prev) * 100, 1) if prev else None,
            "segments": [{"segment": k, "customers": int(v), "revenue": float(g[g.segment == k].spend.sum())} for k, v in g.segment.value_counts().items()],
            "new_vs_returning": [{"month": str(m), "new": int(new[m]), "returning": int(act[m] - new[m])} for m in act.index[-12:]]}

@app.get("/api/geo")
def geo(category: str = None, state: str = None):
    df = D(category, state)
    st = df.groupby("state").agg(revenue=("revenue", "sum"), units=("units", "sum")).sort_values("revenue", ascending=False).reset_index()
    st["share"] = (st.revenue / st.revenue.sum() * 100).round(1)
    ct = df.groupby(["city", "state"]).agg(revenue=("revenue", "sum"), units=("units", "sum")).sort_values("revenue", ascending=False).reset_index()
    return {"states": rec(st), "cities": rec(ct)}

@app.get("/api/insights")
def insights(category: str = None, state: str = None):
    df = D(category, state); out = []
    top = df.groupby("product").revenue.sum().sort_values(ascending=False)
    out.append({"kind": "Revenue", "title": f"{top.index[0]} is your top earner", "text": f"It contributes {top.iloc[0] / top.sum():.0%} of revenue; {top.index[1]} follows." if len(top) > 1 else ""})
    reg = df.groupby("state").revenue.sum().sort_values(ascending=False)
    out.append({"kind": "Region", "title": f"{reg.index[0]} is the best-performing region", "text": f"{reg.iloc[0] / reg.sum():.0%} of revenue. Weakest: {reg.index[-1]} at {reg.iloc[-1] / reg.sum():.0%}."})
    gc = {c: growth(df[df.category == c]) for c in df.category.unique()}; gc = {k: v for k, v in gc.items() if v is not None}
    if gc:
        w = min(gc, key=gc.get); out.append({"kind": "Category", "title": f"{w} is the weakest category", "text": f"Revenue changed {gc[w]:+.1f}% over the last 30 days versus the 30 before."})
    try:
        f = forecast(30, "all", "", category, state); out.append({"kind": "Demand", "title": f"Demand expected to {'rise' if f['trend_pct'] >= 0 else 'fall'} {abs(f['trend_pct'])}%", "text": f"The Random Forest model projects {f['total_units']:,} units over 30 days against the last 28 days' average (backtest MAPE {f['metrics']['random_forest']['mape']}%)."})
    except HTTPException: pass
    rows = inv_rows(df)
    if rows:
        risk = sorted([r for r in rows if r["status"] in ("Low stock", "Out of stock")], key=lambda r: r["days_cover"] or 0)
        out.append({"kind": "Stock risk", "title": f"{len(risk)} product(s) at stockout risk", "text": ", ".join(f"{r['product']} ({r['days_cover']} days)" for r in risk[:4]) or "No product is below 7 days of cover."})
        act = sorted(rows, key=lambda r: -r["reorder_qty"])[:3]
        out.append({"kind": "Action", "title": "Recommended inventory actions", "text": "; ".join(f"Reorder {r['reorder_qty']} × {r['product']}" for r in act if r["reorder_qty"]) or "No reorders needed."})
    return out

# ---------------- data center ----------------
def profile(df):
    num = df.select_dtypes("number")
    return {"rows": len(df), "columns": list(df.columns), "missing": {c: int(v) for c, v in df.isna().sum().items() if v},
            "duplicates": int(df.duplicated().sum()), "preview": json.loads(df.head(10).to_json(orient="records")),
            "stats": json.loads(num.describe().round(2).to_json()) if len(num.columns) else {}}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower(); raw = await file.read()
    if not name.endswith((".csv", ".xlsx", ".xls")): raise HTTPException(400, "Upload a .csv, .xlsx or .xls file.")
    try: df = pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception as e: raise HTTPException(400, f"The file could not be read: {e}")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    miss = [c for c in REQ if c not in df.columns]
    if miss: raise HTTPException(422, f"Missing required columns: {', '.join(miss)}. Required: {', '.join(REQ)}. Optional: order_id, customer_id, store, stock.")
    S["raw"] = df; S["raw_name"] = file.filename
    return profile(df)

@app.post("/api/clean")
def clean():
    raw = S["raw"]
    if raw is None: raise HTTPException(400, "Upload a file first.")
    n0 = len(raw); df = raw.drop_duplicates(); dups = n0 - len(df)
    b = len(df); df = prepare(df); dropped = b - len(df); filled = 0
    for c in df.columns:
        na = int(df[c].isna().sum())
        if not na or c == "stock": continue
        df[c] = df[c].fillna(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else df[c].fillna("Unknown"); filled += na
    if len(df) < 30: raise HTTPException(422, "Fewer than 30 valid rows remain after cleaning.")
    S["df"] = df; S["src"] = S.get("raw_name", "Uploaded dataset"); CACHE.clear()
    return {"rows_before": n0, "duplicates_removed": dups, "invalid_rows_dropped": dropped, "values_filled": filled, "rows_after": len(df)}

@app.post("/api/reset")
def reset(): S["df"] = make_demo(); S["src"] = DEMO; S["raw"] = None; CACHE.clear(); return {"ok": True}

# ---------------- exports ----------------
def csv_resp(df, name):
    return StreamingResponse(io.StringIO(df.to_csv(index=False)), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={name}"})

@app.get("/api/export/{kind}")
def export(kind: str, category: str = None, state: str = None, horizon: int = 30):
    df = D(category, state)
    if kind == "data": return csv_resp(dstr(S["df"]), "doomsters_cleaned_data.csv")
    if kind == "sales": return csv_resp(df.groupby(df.date.dt.strftime("%Y-%m-%d")).agg(revenue=("revenue", "sum"), units=("units", "sum")).reset_index(), "doomsters_sales.csv")
    if kind == "inventory": return csv_resp(pd.DataFrame(inventory(category, state)["rows"]), "doomsters_inventory.csv")
    if kind == "forecast":
        f = forecast(horizon, "all", "", category, state)["series"]
        return csv_resp(pd.DataFrame([{"date": r["date"], "forecast": r["forecast"], "lower": r["band"][0], "upper": r["band"][1]} for r in f if "forecast" in r]), "doomsters_forecast.csv")
    raise HTTPException(404, "Unknown export.")

@app.get("/api/report/pdf")
def pdf(kind: str = "executive", category: str = None, state: str = None):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    ss = getSampleStyleSheet(); o = overview(category, state); buf = io.BytesIO(); el = []
    def tbl(data):
        t = Table(data, hAlign="LEFT"); t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e2a5a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)])); el.extend([t, Spacer(1, 12)])
    el += [Paragraph("DOOMSTERS Retail Intelligence", ss["Title"]), Paragraph(f"{kind.title()} report. Data source: {S['src']}. Filters: {category or 'All'} / {state or 'All'}", ss["Normal"]), Spacer(1, 12)]
    if kind in ("executive", "sales"):
        tbl([["Metric", "Value"], ["Revenue", f"{o['revenue']:,.0f}"], ["Units", f"{o['units']:,}"], ["Orders", f"{o['orders']:,}"], ["Avg order value", f"{o['aov']:,.0f}"], ["30-day growth", f"{o['growth']}%"]])
        s = sales("D", category, state)["products"][:8]; tbl([["Product", "Revenue", "Units", "Share %"]] + [[p["product"], f"{p['revenue']:,.0f}", p["units"], p["share"]] for p in s])
    if kind in ("executive", "forecast"):
        f = forecast(30, "all", "", category, state); m = f["metrics"]
        tbl([["Model", "MAE", "RMSE", "MAPE %"], ["Random Forest", *m["random_forest"].values()], ["7-day moving average", *m["moving_average_7d"].values()]])
        el.append(Paragraph(f"30-day forecast: {f['total_units']:,} units, approx. revenue {f['total_revenue']:,}. Backtested on the last {f['backtest_days']} days.", ss["Normal"])); el.append(Spacer(1, 12))
    if kind in ("executive", "inventory"):
        try: tbl([["Product", "Stock", "Days cover", "Status", "Reorder"]] + [[r["product"], r["stock"], r["days_cover"], r["status"], r["reorder_qty"]] for r in inventory(category, state)["rows"]])
        except HTTPException as e: el.append(Paragraph(e.detail, ss["Normal"]))
    SimpleDocTemplate(buf, pagesize=A4).build(el)
    return Response(buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=doomsters_{kind}_report.pdf"})
