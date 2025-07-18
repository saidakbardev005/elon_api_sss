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

# ─── Fayl yo‘llari ────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
PROJECT_ROOT     = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH       = os.path.join(PROJECT_ROOT, "kmeans_model.pkl")
SCALER_PATH      = os.path.join(PROJECT_ROOT, "scaler.pkl")
PRICE_MODEL_PATH = os.path.join(PROJECT_ROOT, "price_predictor.pkl")

# ─── Google Maps mijozi ───────────────────────────────────────────────────────
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

def load_db_data():
    """
    1) fetch_table("announcements") → pick_up_address, shipping_address, price
       - kolonka nomlarini strip()+lower()+replace() qilamiz
       - pick_col, ship_col, price_col ni fuzzy matching orqali topamiz
       - vergulgacha kesib, transliteratsiya+lowercase qilamiz
       - from_city/to_city deb rename qilamiz
    2) fetch_table bilan driver_locations, my_autos, users jadvallarini olamiz
    """
    # 1) announcements jadvalini yuklaymiz
    ann = fetch_table("announcements")

    # DEBUG: API’dan kelgan kolonka nomlarini ko‘ramiz
    print("[DEBUG] announcements.columns =", ann.columns.tolist())

    # normalize: bosh/oxir bo‘shliq, katta-kichik harfni bartaraf, bo‘shliq va chiziqlarni _
    ann.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in ann.columns
    ]

    # fuzzy matching: pick_up_address, shipping_address, price
    pick_col  = next((c for c in ann.columns if "pick" in c and "address" in c), None)
    ship_col  = next((c for c in ann.columns if "ship" in c and "address" in c), None)
    price_col = next((c for c in ann.columns if c == "price"), None)

    if not all([pick_col, ship_col, price_col]):
        raise KeyError(
            f"Announcements table missing columns: "
            f"{pick_col}, {ship_col}, {price_col}"
        )

    # faqat shu uchta ustunni olamiz
    df_price = ann[[pick_col, ship_col, price_col]].copy()

    # vergulgacha bo‘lgan qismini olib, transliteratsiya + lowercase qilamiz
    for col in [pick_col, ship_col]:
        df_price[col] = (
            df_price[col]
            .astype(str)
            .str.split(",", n=1, expand=True)[0]
            .str.strip()
            .map(lambda x: latin_to_cyrillic(x).strip().lower())
        )

    # rename: pick_col → from_city, ship_col → to_city
    df_price = df_price.rename(columns={
        pick_col: "from_city",
        ship_col: "to_city"
    })

    # faqat kerakli ustunlarni qoldiramiz
    df_price = df_price[["from_city", "to_city", price_col]]

    # 2) qolgan jadvallar
    drivers_df = fetch_table("driver_locations")
    my_autos   = fetch_table("my_autos")
    users      = fetch_table("users")

    # users.id → user_id qilib rename qilamiz
    if "id" in users.columns:
        users = users.rename(columns={"id": "user_id"})
    users = users[["user_id", "fullname", "phone", "status"]]

    return df_price, drivers_df, my_autos, users

def get_coordinates(location: str):
    """Google Maps Geocoding API yordamida (lat, lng) qaytaradi."""
    try:
        res = gmaps.geocode(location)
        if res:
            loc = res[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None

def load_or_initialize_model():
    """KMeans + Scaler ni yuklash yoki yaratish."""
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
    """Yangi yuk uchun klaster bashorati + modelni saqlash."""
    global model, scaler
    Xs = scaler.transform([[weight, volume]])
    model.partial_fit(Xs)
    joblib.dump(model, MODEL_PATH)
    return int(model.predict(Xs)[0])

def load_or_initialize_price_model():
    """SGDRegressor modelini yuklash yoki yaratish."""
    if os.path.exists(PRICE_MODEL_PATH):
        return joblib.load(PRICE_MODEL_PATH)
    pm = SGDRegressor()
    pm.partial_fit([[0, 0]], [0])
    joblib.dump(pm, PRICE_MODEL_PATH)
    return pm

def online_fit_and_predict_price(f_enc: int, t_enc: int, actual_price: float=None) -> int:
    """Viloyat kodlari bo‘yicha online narx bashorati + yangilash."""
    pm = load_or_initialize_price_model()
    X  = [[f_enc, t_enc]]
    if actual_price is not None:
        pm.partial_fit(X, [actual_price])
        joblib.dump(pm, PRICE_MODEL_PATH)
    return max(0, int(pm.predict(X)[0]))

def train_price_model_from_db():
    """Batch rejimida announcements’dan narx modelini o‘rgatish."""
    try:
        df_price, _, _, _ = load_db_data()
    except Exception as e:
        print(f"❌ Price model uchun ma’lumot yuklanmadi: {e}")
        return

    regions = df_price["from_city"].tolist() + df_price["to_city"].tolist()
    le = LabelEncoder().fit(regions)

    X, y = [], []
    for row in df_price.itertuples(index=False):
        f_enc = le.transform([row.from_city])[0]
        t_enc = le.transform([row.to_city])[0]
        p     = float(getattr(row, price_col)) if hasattr(row, price_col) else float(row.price)
        if p > 0:
            X.append([f_enc, t_enc])
            y.append(p)

    if not X:
        print("❌ Valid narx ma’lumot topilmadi.")
        return

    pm = SGDRegressor()
    pm.partial_fit(X, y)
    joblib.dump(pm, PRICE_MODEL_PATH)
    print(f"✅ Narx modeli {len(X)} namunada o‘qitildi.")

def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    """
    Eng mos 5 haydovchini qaytaradi:
     - sig‘im bo‘yicha klasterlash
     - sig‘im & masofa bo‘yicha saralash
    """
    try:
        _, drivers_df, my_autos, users = load_db_data()
    except Exception as e:
        print(f"❌ Haydovchilar ma’lumot yuklanmadi: {e}")
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
    same = df[df['cluster_id']==cid].copy()
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
