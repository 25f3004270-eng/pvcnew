#!/usr/bin/env python3
import os
import json
import math
from datetime import datetime, date

import pandas as pd
from dateutil.relativedelta import relativedelta

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from waitress import serve
from functools import wraps


# =========================================================
# CONFIG
# =========================================================
GST_FACTOR = 1.18

# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)

# Secret key from environment (safe for production)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "fallback-secret-key")

# Database configuration (Render PostgreSQL)
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# =========================================================
# MODELS
# =========================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)


class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False)
    code = db.Column(db.String(50), unique=True)
    description = db.Column(db.Text)

    pvc_formula_code = db.Column(db.String(50), nullable=False)
    weights_json = db.Column(db.Text)          # JSON of index weights
    extra_fields_json = db.Column(db.Text)     # JSON for dynamic extra inputs


class ItemIndex(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # which item this index row belongs to
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    item = db.relationship('Item')

    # month for which these indices apply (store 1st of the month)
    month = db.Column(db.Date, nullable=False)

    # JSON with all component indices for this item & month
    # example: {"copper": 865.2, "crgo": 710.0, "ms": 130.5}
    # or {"C": 345, "S": 290, "IS": 150, "PV": 210, "W": 180}
    indices_json = db.Column(db.Text, nullable=False)


class PVCResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User')

    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    item = db.relationship('Item')

    username = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    basicrate = db.Column(db.Float)
    quantity = db.Column(db.Float)
    freight = db.Column(db.Float)

    pvcbasedate = db.Column(db.String(10))
    origdp = db.Column(db.String(10))
    refixeddp = db.Column(db.String(10))
    extendeddp = db.Column(db.String(10))
    caldate = db.Column(db.String(10))
    supdate = db.Column(db.String(10))
    rateapplied = db.Column(db.String(50))

    pvcactual = db.Column(db.Float)
    pvccontractual = db.Column(db.Float)
    ldamtactual = db.Column(db.Float)
    ldamtcontractual = db.Column(db.Float)
    fairprice = db.Column(db.Float)
    selectedscenario = db.Column(db.String(10))


# =========================================================
# LOGIN + ADMIN HELPER
# =========================================================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def admin_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()
        if not getattr(current_user, "is_admin", False):
            abort(403)
        return view_func(*args, **kwargs)
    return wrapped_view


# =========================================================
# UTILS: ITEM INDEX DATAFRAME + PVC % CALC
# =========================================================
def get_item_index_df(item):
    """Load per-item indices into a Pandas DataFrame indexed by month."""
    rows = ItemIndex.query.filter_by(item_id=item.id).order_by(ItemIndex.month).all()
    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        if not r.month:
            continue
        try:
            indices = json.loads(r.indices_json or "{}")
        except Exception:
            indices = {}
        row_dict = {"date": pd.Timestamp(r.month.year, r.month.month, 1)}
        row_dict.update(indices)
        data.append(row_dict)

    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df.set_index("date").sort_index()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_round(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return None


def to_month_start(d):
    d = pd.to_datetime(d, errors="coerce")
    if pd.isna(d):
        return pd.NaT
    return pd.Timestamp(d.year, d.month, 1)


def previous_month(d):
    d = to_month_start(d)
    if pd.isna(d):
        return pd.NaT
    return d - relativedelta(months=1)


def ieema_row(df, date, previous=False):
    if df is None or df.empty:
        return None
    if pd.isna(date):
        return None
    target = previous_month(date) if previous else to_month_start(date)
    eligible = df[df.index <= target]
    return eligible.iloc[-1] if not eligible.empty else None


def pvc_percent(base_date, current_date, idx_df, weights):
    base = ieema_row(idx_df, base_date, previous=False)
    curr = ieema_row(idx_df, current_date, previous=True)
    if base is None or curr is None:
        return 0

    total = 0.0
    for k, w in weights.items():
        b = base.get(k)
        c = curr.get(k)
        if b and c:
            total += w * ((c - b) / b)
    return total


def pvc_percent_detailed(base_date, current_date, idx_df, scenario, weights):
    base = ieema_row(idx_df, base_date, previous=False)
    curr = ieema_row(idx_df, current_date, previous=True)
    if base is None or curr is None:
        return None

    row = {
        "scenario": scenario,
        "base_month": base.name,
        "current_month": curr.name,
    }

    total = 0.0
    for k, w in weights.items():
        b = base.get(k)
        c = curr.get(k)
        if b and c:
            contrib = round(w * ((c - b) / b), 4)
            total += contrib
        else:
            contrib = None

        row[f"{k}_base"] = round(b, 2) if b is not None else None
        row[f"{k}_current"] = round(c, 2) if c is not None else None
        row[f"{k}_weight"] = w
        row[f"{k}_contribution_pct"] = contrib

    row["pvc_percent"] = round(total, 4)
    return row


# =========================================================
# CORE SINGLE-RECORD PVC + LD CALC
# =========================================================
def calculate_single_record_from_dict(d, idx_df, weights):
    pvc_base_date = pd.to_datetime(d.get("pvc_base_date"), errors="coerce")
    call_date = pd.to_datetime(d.get("call_date"), errors="coerce")

    orig_dp = pd.to_datetime(d.get("orig_dp"), errors="coerce")
    refixed_dp = pd.to_datetime(d.get("refixeddp"), errors="coerce")
    extended_dp = pd.to_datetime(d.get("extendeddp"), errors="coerce")

    scheduled_date = refixed_dp if pd.notna(refixed_dp) else orig_dp

    sup_date = pd.to_datetime(d.get("sup_date"), errors="coerce")
    lower_basic_date = pd.to_datetime(d.get("lower_basic_date"), errors="coerce")

    acc_qty = safe_float(d.get("acc_qty"))
    basic_rate = safe_float(d.get("basic_rate"))
    freight_rate = safe_float(d.get("freight_rate_per_unit"))

    lower_rate = safe_float(d.get("lower_rate"))
    lower_freight = safe_float(d.get("lower_freight"))

    freight = freight_rate * acc_qty

    pct_a2 = pvc_percent(pvc_base_date, call_date, idx_df, weights)
    pct_b2 = pvc_percent(pvc_base_date, scheduled_date, idx_df, weights)
    pct_c1 = pvc_percent(lower_basic_date, call_date, idx_df, weights)
    pct_d1 = pvc_percent(lower_basic_date, scheduled_date, idx_df, weights)

    pvc_ps_a2 = basic_rate * pct_a2 / 100 if pct_a2 is not None else 0
    pvc_ps_b2 = basic_rate * pct_b2 / 100 if pct_b2 is not None else 0
    pvc_ps_c1 = lower_rate * pct_c1 / 100 if pct_c1 is not None else 0
    pvc_ps_d1 = lower_rate * pct_d1 / 100 if pct_d1 is not None else 0

    base_amt = basic_rate * acc_qty
    lower_amt = lower_rate * acc_qty
    lower_freight_total = lower_freight * acc_qty

    pvc_actual = (
        (base_amt + base_amt * pct_a2 / 100 + freight) * GST_FACTOR
        if pct_a2 is not None else 0
    )
    pvc_contractual = (
        (base_amt + base_amt * pct_b2 / 100 + freight) * GST_FACTOR
        if pct_b2 is not None else 0
    )
    lower_actual = (
        (lower_amt + lower_amt * pct_c1 / 100 + lower_freight_total) * GST_FACTOR
        if pct_c1 is not None else 0
    )
    lower_contractual = (
        (lower_amt + lower_amt * pct_d1 / 100 + lower_freight_total) * GST_FACTOR
        if pct_d1 is not None else 0
    )

    # ----- LD logic -----
    delay_days = 0
    ld_weeks_new = 0
    ld_rate_pct_new = 0
    ld_applicable = True
    ld_base = None

    if pd.notna(extended_dp):
        # extended without LD – LD applies only further extension case; here treat as no LD
        ld_applicable = False
        ld_base = extended_dp
    elif pd.notna(refixed_dp):
        ld_base = refixed_dp
    else:
        ld_base = orig_dp

    if ld_applicable and pd.notna(sup_date) and pd.notna(ld_base):
        delay_days = max((sup_date - ld_base).days, 0)
    else:
        delay_days = 0

    if not ld_applicable or delay_days <= 0:
        ld_applicable = False
        delay_days = 0
        ld_weeks_new = 0
        ld_rate_pct_new = 0
    else:
        ld_weeks_new = math.ceil(delay_days / 7)
        ld_rate_pct_new = min(ld_weeks_new * 0.5, 10)

    ld_amt_actual = max(pvc_actual, 0) * ld_rate_pct_new / 100
    ld_amt_contractual = max(pvc_contractual, 0) * ld_rate_pct_new / 100

    pvc_actual_less_ld_new = (pvc_actual - ld_amt_actual if pvc_actual else None)
    pvc_contractual_less_ld_new = (
        pvc_contractual - ld_amt_contractual if pvc_contractual else None
    )

    lower_actual_less_ld = (lower_actual - ld_amt_actual if lower_actual else None)
    lower_contractual_less_ld = (
        lower_contractual - ld_amt_contractual if lower_contractual else None
    )

    rateapplied = str(d.get("rateapplied", "")).strip().lower()

    if rateapplied == "supply before due date":
        candidates_new = {"A2": pvc_actual}
    elif rateapplied == "supply after due date":
        candidates_new = {"A2": pvc_actual, "B2": pvc_contractual}
    elif rateapplied == "lower rate applicable":
        candidates_new = {
            "A2": pvc_actual,
            "B2": pvc_contractual,
            "C1": lower_actual,
            "D1": lower_contractual,
        }
    elif rateapplied == "lower rate and ld comparison":
        candidates_new = {
            "A2": pvc_actual_less_ld_new,
            "B2": pvc_contractual_less_ld_new,
            "C1": lower_actual,
            "D1": lower_contractual,
        }
    elif rateapplied == "lower rate with ld in further extension":
        candidates_new = {
            "A2": pvc_actual_less_ld_new,
            "B2": pvc_contractual_less_ld_new,
            "C1": lower_actual_less_ld,
            "D1": lower_contractual_less_ld,
        }
    else:
        candidates_new = {
            "A2": pvc_actual,
            "B2": pvc_contractual,
            "C1": lower_actual,
            "D1": lower_contractual,
        }

    candidates_new = {k: v for k, v in candidates_new.items() if v is not None}
    selected_scenario_new = (
        min(candidates_new, key=candidates_new.get) if candidates_new else None
    )
    fair_price_new = candidates_new.get(selected_scenario_new, 0)

    result_row = {
        "acc_qty": acc_qty,
        "basic_rate": basic_rate,
        "pvc_base_date": pvc_base_date,
        "lower_rate": lower_rate,
        "lower_freight": lower_freight,
        "lower_freight_total": safe_round(lower_freight_total),
        "lower_basic_date": lower_basic_date,
        "freight_rate_per_unit": freight_rate,
        "freight": safe_round(freight),
        "orig_dp": orig_dp,
        "refixeddp": refixed_dp,
        "extendeddp": extended_dp,
        "scheduled_date": scheduled_date,
        "call_date": call_date,
        "sup_date": sup_date,
        "pvc_actual": safe_round(pvc_actual),
        "pvc_contractual": safe_round(pvc_contractual),
        "lower_actual": safe_round(lower_actual),
        "lower_contractual": safe_round(lower_contractual),
        "delay_days": delay_days,
        "ld_weeks_new": ld_weeks_new,
        "ld_rate_pct_new": safe_round(ld_rate_pct_new),
        "ld_applicable": ld_applicable,
        "ld_amt_actual": safe_round(ld_amt_actual),
        "ld_amt_contractual": safe_round(ld_amt_contractual),
        "pvc_actual_less_ld_new": safe_round(pvc_actual_less_ld_new),
        "pvc_contractual_less_ld_new": safe_round(pvc_contractual_less_ld_new),
        "lower_actual_less_ld": safe_round(lower_actual_less_ld),
        "lower_contractual_less_ld": safe_round(lower_contractual_less_ld),
        "fair_price_new": safe_round(fair_price_new),
        "selected_scenario_new": selected_scenario_new,
        "pvc_per_set_a2": pvc_ps_a2,
        "pvc_per_set_b2": pvc_ps_b2,
        "pvc_per_set_c1": pvc_ps_c1,
        "pvc_per_set_d1": pvc_ps_d1,
    }

    scenario_amounts = {
        "A2": safe_round(pvc_actual_less_ld_new),
        "B2": safe_round(pvc_contractual_less_ld_new),
        "C1": safe_round(lower_actual),
        "D1": safe_round(lower_contractual),
    }
    result_row["scenario_amounts"] = scenario_amounts

    scenario_details = []
    for sc, bd, cd in [
        ("A2", pvc_base_date, call_date),
        ("B2", pvc_base_date, scheduled_date),
        ("C1", lower_basic_date, call_date),
        ("D1", lower_basic_date, scheduled_date),
    ]:
        det = pvc_percent_detailed(bd, cd, idx_df, sc, weights)
        if det:
            scenario_details.append(det)
    result_row["scenario_details"] = scenario_details

    return result_row


def calculate_for_item(item, one, index_df, weights):
    """Dispatcher if you later need different logic per pvc_formula_code."""
    code = (item.pvc_formula_code or "").upper()

    # For now all formulas reuse the same engine; you can branch here later.
    if code in ("TM_IEEMA_RM", "POWER_TRF_IEEMA"):
        return calculate_single_record_from_dict(one, index_df, weights)
    else:
        return calculate_single_record_from_dict(one, index_df, weights)


# =========================================================
# AUTH ROUTES
# =========================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Username and password required.", "danger")
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("User already exists!", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        user = User.query.filter_by(username=u).first()
        if user and check_password_hash(user.password_hash, p):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# =========================================================
# MAIN USER ROUTES
# =========================================================
@app.route('/')
@login_required
def index():
    items = Item.query.order_by(Item.name).all()
    return render_template('index.html', items=items)


@app.route('/calculate', methods=['POST'])
@login_required
def calculate():
    item_id = request.form.get('item_id')
    if not item_id:
        flash("Please select an item.", "danger")
        return redirect(url_for('index'))

    item = Item.query.get_or_404(int(item_id))

    # Indices DF from DB for this item
    index_df = get_item_index_df(item)
    if index_df.empty:
        flash("Indices not found in database for this item. Please configure in admin.", "danger")
        return redirect(url_for('index'))

    try:
        weights = json.loads(item.weights_json or "{}")
    except Exception:
        weights = {}

    # standard fields
    origdp_str = request.form.get('origdp') or ''
    refixeddp_str = request.form.get('refixeddp') or ''
    extendeddp_str = request.form.get('extendeddp') or ''
    supply_str = request.form.get('supdate') or ''
    rateapplied = request.form.get('rateapplied') or ''

    data = {
        'user_id': current_user.id,
        'username': current_user.username,
        'item_id': item.id,
        'item_name': item.name,

        'basicrate': float(request.form.get('basicrate', 0) or 0),
        'quantity': float(request.form.get('quantity', 0) or 0),
        'freight': float(request.form.get('freight', 0) or 0),

        'pvcbasedate': request.form.get('pvcbasedate') or '',
        'origdp': origdp_str,
        'refixeddp': refixeddp_str,
        'extendeddp': extendeddp_str,
        'caldate': request.form.get('caldate') or '',
        'supdate': supply_str,
        'rateapplied': rateapplied,

        'lowerrate': float(request.form.get('lowerrate', 0) or 0),
        'lowerfreight': float(request.form.get('lowerfreight', 0) or 0),
        'lowerbasicdate': request.form.get('lowerbasicdate') or '',
    }

    # map to calculator dict
    one = {
        "acc_qty": data['quantity'],
        "basic_rate": data['basicrate'],
        "freight_rate_per_unit": data['freight'],

        "pvc_base_date": data['pvcbasedate'],
        "call_date": data['caldate'],

        "orig_dp": data['origdp'],
        "refixeddp": data['refixeddp'],
        "extendeddp": data['extendeddp'],

        "sup_date": data['supdate'],

        "lower_rate": data['lowerrate'],
        "lower_freight": data['lowerfreight'],
        "lower_basic_date": data['lowerbasicdate'],
        "rateapplied": data['rateapplied'],
    }

    try:
        result_row = calculate_for_item(item, one, index_df, weights)
    except Exception as e:
        app.logger.exception("PVC calculation failed")
        flash("PVC calculation failed. Please check your inputs.", "danger")
        return redirect(url_for('index'))

    scenario_amounts = result_row.get("scenario_amounts", {})
    selected = result_row.get("selected_scenario_new")

    if data["rateapplied"].strip().lower() == "supply before due date":
        selected = "A2"

    result = {
        "data": {
            "pvcactual": result_row.get("pvc_actual", 0.0),
            "pvccontractual": result_row.get("pvc_contractual", 0.0),
            "lower_actual": result_row.get("lower_actual", 0.0),
            "lower_contractual": result_row.get("lower_contractual", 0.0),
            "ldamtactual": result_row.get("ld_amt_actual", 0.0),
            "ldamtcontractual": result_row.get("ld_amt_contractual", 0.0),

            "fairprice": result_row.get("fair_price_new", 0.0),

            "pvc_actual_less_ld_new": result_row.get("pvc_actual_less_ld_new"),
            "pvc_contractual_less_ld_new": result_row.get("pvc_contractual_less_ld_new"),
            "lower_actual_less_ld": result_row.get("lower_actual_less_ld"),
            "lower_contractual_less_ld": result_row.get("lower_contractual_less_ld"),

            "delay_days": result_row.get("delay_days"),
            "ld_weeks": result_row.get("ld_weeks_new"),
            "ld_rate_pct": result_row.get("ld_rate_pct_new"),
            "ld_applicable": result_row.get("ld_applicable", True),
            "selectedscenario": selected,

            "pvc_per_set_a2": result_row.get("pvc_per_set_a2"),
            "pvc_per_set_b2": result_row.get("pvc_per_set_b2"),
            "pvc_per_set_c1": result_row.get("pvc_per_set_c1"),
            "pvc_per_set_d1": result_row.get("pvc_per_set_d1"),
        },
        "scenario_details": result_row.get("scenario_details", []),
        "scenario_amounts": scenario_amounts,
    }

    calc = PVCResult(
        user_id=data['user_id'],
        username=data['username'],
        item_id=item.id,
        basicrate=data['basicrate'],
        quantity=data['quantity'],
        freight=data['freight'],
        pvcbasedate=data['pvcbasedate'],
        origdp=data['origdp'],
        refixeddp=data['refixeddp'],
        extendeddp=data['extendeddp'],
        caldate=data['caldate'],
        supdate=data['supdate'],
        rateapplied=data['rateapplied'],
        pvcactual=result["data"]["pvcactual"],
        pvccontractual=result["data"]["pvccontractual"],
        ldamtactual=result["data"]["ldamtactual"],
        ldamtcontractual=result["data"]["ldamtcontractual"],
        fairprice=result["data"]["fairprice"],
        selectedscenario=result["data"]["selectedscenario"],
    )
    db.session.add(calc)
    db.session.commit()

    return render_template(
        'result.html',
        item=item.name,
        item_obj=item,
        data=data,
        result=result,
        calc_id=calc.id
    )


@app.route('/history')
@login_required
def history():
    records = PVCResult.query.filter_by(user_id=current_user.id) \
                             .order_by(PVCResult.created_at.desc()) \
                             .all()
    return render_template('history.html', records=records)


@app.route('/calc/<int:calc_id>')
@login_required
def view_calc(calc_id):
    calc = PVCResult.query.filter_by(id=calc_id, user_id=current_user.id).first_or_404()
    item = calc.item

    data = {
        "basicrate": calc.basicrate,
        "quantity": calc.quantity,
        "freight": calc.freight,
        "pvcbasedate": calc.pvcbasedate,
        "origdp": calc.origdp,
        "refixeddp": calc.refixeddp,
        "extendeddp": calc.extendeddp,
        "caldate": calc.caldate,
        "supdate": calc.supdate,
        "rateapplied": calc.rateapplied,
    }

    result = {
        "data": {
            "pvcactual": calc.pvcactual,
            "pvccontractual": calc.pvccontractual,
            "ldamtactual": calc.ldamtactual,
            "ldamtcontractual": calc.ldamtcontractual,
            "fairprice": calc.fairprice,
            "selectedscenario": calc.selectedscenario,
            "ld_applicable": True if calc.ldamtactual and calc.ldamtactual > 0 else False,
        },
        "scenario_details": [],
        "scenario_amounts": {},
    }

    return render_template(
        'result.html',
        item=item.name,
        item_obj=item,
        data=data,
        result=result,
        calc_id=calc.id
    )


# =========================================================
# SIMPLE EXCEL EXPORT FOR ONE CALC
# =========================================================
@app.route('/calc/<int:calc_id>/excel')
@login_required
def export_calc_excel(calc_id):
    calc = PVCResult.query.filter_by(id=calc_id, user_id=current_user.id).first_or_404()
    item = calc.item

    df = pd.DataFrame([{
        "Calc ID": calc.id,
        "User": calc.username,
        "Item": item.name if item else "",
        "Basic Rate": calc.basicrate,
        "Quantity": calc.quantity,
        "Freight/Unit": calc.freight,
        "PVC Base Date": calc.pvcbasedate,
        "Original DP": calc.origdp,
        "Refixed DP": calc.refixeddp,
        "Extended DP": calc.extendeddp,
        "Call Date": calc.caldate,
        "Supply Date": calc.supdate,
        "Rate Applied": calc.rateapplied,
        "PVC Actual": calc.pvcactual,
        "PVC Contractual": calc.pvccontractual,
        "LD Actual": calc.ldamtactual,
        "LD Contractual": calc.ldamtcontractual,
        "Fair Price": calc.fairprice,
        "Selected Scenario": calc.selectedscenario,
    }])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='PVC_RESULT')
    output.seek(0)

    fname = f"PVC_{calc.id}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# ADMIN: ITEMS CRUD (ADMIN ONLY)
# =========================================================
@app.route('/admin/items')
@login_required
@admin_required
def admin_items_list():
    items = Item.query.order_by(Item.name).all()
    return render_template('admin_items_list.html', items=items)


@app.route('/admin/items/new', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_items_new():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        code = request.form.get('code', '').strip()
        formula = request.form.get('pvc_formula_code', '').strip()
        weights_json = request.form.get('weights_json', '').strip() or "{}"
        extra_fields_json = request.form.get('extra_fields_json', '').strip() or "[]"

        if not name or not formula:
            flash("Name and formula code are required.", "danger")
            return redirect(url_for('admin_items_new'))

        try:
            json.loads(weights_json)
            json.loads(extra_fields_json)
        except Exception:
            flash("Weights/Extra fields must be valid JSON.", "danger")
            return redirect(url_for('admin_items_new'))

        it = Item(
            name=name,
            code=code or None,
            pvc_formula_code=formula,
            weights_json=weights_json,
            extra_fields_json=extra_fields_json,
            description=request.form.get('description', '')
        )
        db.session.add(it)
        db.session.commit()
        flash("Item added.", "success")
        return redirect(url_for('admin_items_list'))

    return render_template('admin_items_form.html', item=None)


@app.route('/admin/items/<int:item_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_items_edit(item_id):
    it = Item.query.get_or_404(item_id)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        code = request.form.get('code', '').strip()
        formula = request.form.get('pvc_formula_code', '').strip()
        weights_json = request.form.get('weights_json', '').strip() or "{}"
        extra_fields_json = request.form.get('extra_fields_json', '').strip() or "[]"

        if not name or not formula:
            flash("Name and formula code are required.", "danger")
            return redirect(url_for('admin_items_edit', item_id=it.id))

        try:
            json.loads(weights_json)
            json.loads(extra_fields_json)
        except Exception:
            flash("Weights/Extra fields must be valid JSON.", "danger")
            return redirect(url_for('admin_items_edit', item_id=it.id))

        it.name = name
        it.code = code or None
        it.pvc_formula_code = formula
        it.weights_json = weights_json
        it.extra_fields_json = extra_fields_json
        it.description = request.form.get('description', '')

        db.session.commit()
        flash("Item updated.", "success")
        return redirect(url_for('admin_items_list'))

    return render_template('admin_items_form.html', item=it)


# =========================================================
# ADMIN: ITEM-SPECIFIC INDICES (ADMIN ONLY)
# =========================================================
@app.route('/admin/items/<int:item_id>/indices')
@login_required
@admin_required
def admin_item_indices_list(item_id):
    item = Item.query.get_or_404(item_id)
    rows = ItemIndex.query.filter_by(item_id=item.id) \
                          .order_by(ItemIndex.month.desc()) \
                          .all()
    return render_template('admin_item_indices_list.html', item=item, rows=rows)


@app.route('/admin/items/<int:item_id>/indices/new', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_item_indices_new(item_id):
    item = Item.query.get_or_404(item_id)

    if request.method == 'POST':
        try:
            month_str = request.form.get('month')
            m = datetime.strptime(month_str, '%Y-%m-%d').date()
            m = date(m.year, m.month, 1)
        except Exception:
            flash("Invalid month date.", "danger")
            return redirect(url_for('admin_item_indices_new', item_id=item.id))

        indices_json = (request.form.get('indices_json') or "").strip()
        try:
            json.loads(indices_json)
        except Exception:
            flash("Indices JSON must be valid JSON.", "danger")
            return redirect(url_for('admin_item_indices_new', item_id=item.id))

        row = ItemIndex(item_id=item.id, month=m, indices_json=indices_json)
        db.session.add(row)
        db.session.commit()
        flash("Indices month added for this item.", "success")
        return redirect(url_for('admin_item_indices_list', item_id=item.id))

    return render_template('admin_item_indices_form.html', item=item, row=None)


@app.route('/admin/items/<int:item_id>/indices/<int:row_id>/edit',
           methods=['GET', 'POST'])
@login_required
@admin_required
def admin_item_indices_edit(item_id, row_id):
    item = Item.query.get_or_404(item_id)
    row = ItemIndex.query.filter_by(id=row_id, item_id=item.id).first_or_404()

    if request.method == 'POST':
        try:
            month_str = request.form.get('month')
            m = datetime.strptime(month_str, '%Y-%m-%d').date()
            row.month = date(m.year, m.month, 1)
        except Exception:
            flash("Invalid month date.", "danger")
            return redirect(url_for('admin_item_indices_edit',
                                    item_id=item.id, row_id=row.id))

        indices_json = (request.form.get('indices_json') or "").strip()
        try:
            json.loads(indices_json)
        except Exception:
            flash("Indices JSON must be valid JSON.", "danger")
            return redirect(url_for('admin_item_indices_edit',
                                    item_id=item.id, row_id=row.id))

        row.indices_json = indices_json
        db.session.commit()
        flash("Indices month updated for this item.", "success")
        return redirect(url_for('admin_item_indices_list', item_id=item.id))

    return render_template('admin_item_indices_form.html', item=item, row=row)


# =========================================================
# INIT DB AND RUN
# =========================================================
def init_db():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123'),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created: admin / admin123")

    # seed 1–2 items if table empty
    if Item.query.count() == 0:
        default_weights = {
            "copper": 40,
            "crgo": 24,
            "ms": 8,
            "insmat": 4,
            "transoil": 8,
            "wpi": 8,
        }
        itm = Item(
            name="Main Transformer 6531 KVA (PL NO: 29721008)",
            code="TRANSFORMER_6531",
            pvc_formula_code="POWER_TRF_IEEMA",
            weights_json=json.dumps(default_weights),
            extra_fields_json="[]",
            description="Default transformer PVC formula using item indices."
        )
        db.session.add(itm)
        db.session.commit()
        print("Seeded default item: Main Transformer 6531 KVA")


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    with app.app_context():
        init_db()
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
