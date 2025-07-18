import os
import joblib
import numpy as np
import pandas as pd
import googlemaps
import warnings

from config import Config
from services.db import fetch_table
from transliteration.latin_to_cyrillic import latin_to_cyrillic
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ─── Ижро учун файл йўллари ─────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
PROJECT_ROOT     = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH       = os.path.join(PROJECT_ROOT, "kmeans_model.pkl")
SCALER_PATH      = os.path.join(PROJECT_ROOT, "scaler.pkl")
PRICE_MODEL_PATH = os.path.join(PROJECT_ROOT, "price_predictor.pkl")

# ─── Google Maps мижозини яратиш ───────────────────────────────────────────────
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

def load_db_data():
    """
    1) announcements → pick_up_address, shipping_address, price
       - колонка номларини кичик ҳарф ва бўшлиқсиз қиламиз
       - биринчи вордни (вергулгича) олиб, кирилл+lowercase
       - from_city/to_city номлари билан қайта номлаймиз
    2) driver_locations, my_autos, users → тўлиқ DataFrame
    """
    # 1) announcements жадвалини оламиз
    ann = fetch_table("announcements")
    # колонка номларини кичик ҳарф ва бош/охир бўшлиқсиз қиламиз
    ann.columns = [c.strip().lower() for c in ann.columns]

    # керакли учта устун борлигини текшириш
    required = ["pick_up_address", "shipping_address", "price"]
    if not all(col in ann.columns for col in required):
        raise KeyError(f"Announcements table missing columns: {required}")

    df_price = ann[required].copy()

    # 2) Вергулгича бўлган йўлни кесиб олиш + транслитерация + lowercase
    for col in ["pick_up_address", "shipping_address"]:
        df_price[col] = (
            df_price[col]
            .astype(str)
            .str.split(",", n=1, expand=True)[0]
            .str.strip()
            .map(lambda x: latin_to_cyrillic(x).strip().lower())
        )

    # 3) from_city / to_city деб қайта номлаймиз
    df_price = df_price.rename(columns={
        "pick_up_address": "from_city",
        "shipping_address": "to_city"
    })

    # 4) фақат керакли устунлар
    df_price = df_price[["from_city", "to_city", "price"]]

    # 5) қолган жадвалларни олиш
    drivers_df = fetch_table("driver_locations")
    my_autos   = fetch_table("my_autos")

    users = fetch_table("users")
    if "id" in users.columns:
        users = users.rename(columns={"id": "user_id"})
    users = users[["user_id", "fullname", "phone", "status"]]

    return df_price, drivers_df, my_autos, users

def get_coordinates(location: str):
    """Google Maps Geocoding API: жойлашув → (lat, lng)."""
    try:
        res = gmaps.geocode(location)
        if res:
            loc = res[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None

def load_or_initialize_model():
    """Clustering модели ва scaler бор-ёқлигини текширади, ёқ бўлса яратади."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        dummy  = np.array([[1000, 10], [2000, 20]])
        scaler = StandardScaler().fit(dummy)
        model  = MiniBatchKMeans(n_clusters=2, batch_size=2, random_state=42)
        model.partial_fit(scaler.transform(dummy))
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler

model, scaler = load_or_initialize_model()

def online_fit_and_predict(weight: float, volume: float) -> int:
    """Янги weight/volume → кластерни башорат қилади ва модельни сақлайди."""
    global model, scaler
    Xs = scaler.transform([[weight, volume]])
    model.partial_fit(Xs)
    joblib.dump(model, MODEL_PATH)
    return int(model.predict(Xs)[0])

def load_or_initialize_price_model():
    """SGDRegressor модели бор-ёқлигини текширади, йўқ бўлса яратади."""
    if os.path.exists(PRICE_MODEL_PATH):
        return joblib.load(PRICE_MODEL_PATH)
    pm = SGDRegressor()
    pm.partial_fit([[0, 0]], [0])
    joblib.dump(pm, PRICE_MODEL_PATH)
    return pm

def online_fit_and_predict_price(f_enc: int, t_enc: int, actual_price: float = None) -> int:
    """Кодланган вилоятлар бўйича онлайн(price) материаллаштириш."""
    pm = load_or_initialize_price_model()
    X  = [[f_enc, t_enc]]
    if actual_price is not None:
        pm.partial_fit(X, [actual_price])
        joblib.dump(pm, PRICE_MODEL_PATH)
    pred = pm.predict(X)[0]
    return max(0, int(pred))

def train_price_model_from_db():
    """Batch-режимида announcements’дан price-моделни ўрганади."""
    try:
        df_price, _, _, _ = load_db_data()
    except Exception as e:
        print(f"❌ Could not load DB data for price model: {e}")
        return

    regions = df_price["from_city"].tolist() + df_price["to_city"].tolist()
    le = LabelEncoder().fit(regions)

    X, y = [], []
    for row in df_price.itertuples(index=False):
        f_enc = le.transform([row.from_city])[0]
        t_enc = le.transform([row.to_city])[0]
        p     = float(row.price)
        if p > 0:
            X.append([f_enc, t_enc])
            y.append(p)

    if not X:
        print("❌ No valid price data found.")
        return

    pm = SGDRegressor()
    pm.partial_fit(X, y)
    joblib.dump(pm, PRICE_MODEL_PATH)
    print(f"✅ Price model trained on {len(X)} samples.")

def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    """
    Eng мос 5 та haydovchini топади:
      - кластерлаш бўйича
      - sig‘им ва гео-масофага қараб сарлайди
    """
    try:
        _, drivers_df, my_autos, users = load_db_data()
    except Exception as e:
        print(f"❌ Could not load DB data for drivers: {e}")
        return []

    drivers_df['latitude']       = pd.to_numeric(drivers_df['latitude'], errors='coerce')
    drivers_df['longitude']      = pd.to_numeric(drivers_df['longitude'], errors='coerce')
    my_autos['transport_weight'] = pd.to_numeric(my_autos['transport_weight'], errors='coerce')
    my_autos['transport_volume'] = pd.to_numeric(my_autos['transport_volume'], errors='coerce')

    df = (
        drivers_df
        .merge(my_autos, on='user_id')
        .merge(users, on='user_id')
        .dropna(subset=['transport_weight','transport_volume','latitude','longitude'])
        .copy()
    )

    arr = df[['transport_weight','transport_volume']].values
    if arr.size == 0:
        return []

    scaler_loc = StandardScaler().fit(arr)
    arr_s      = scaler_loc.transform(arr)
    km         = MiniBatchKMeans(n_clusters=min(4, len(arr_s)), batch_size=6, random_state=42)
    km.partial_fit(arr_s)

    cid = int(km.predict(scaler_loc.transform([[weight, volume]]))[0])
    df['cluster_id'] = km.predict(arr_s)
    same = df[df['cluster_id'] == cid].copy()
    if same.empty:
        return []

    same['capacity_distance'] = np.hypot(
        same['transport_weight'] - weight,
        same['transport_volume'] - volume
    )
    same['distance_km'] = (
        np.hypot(
            same['latitude'] - lat,
            same['longitude'] - lon
        ) * 111
    )

    return (
        same
        .sort_values(by=['capacity_distance','distance_km'])
        .head(5)[[
            'fullname','phone','transport_model',
            'transport_weight','transport_volume','distance_km'
        ]]
        .to_dict(orient='records')
    )
