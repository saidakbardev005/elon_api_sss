# services/predict_service.py

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

# Loyihangiz papkasi va model fayllari manzillari
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH       = os.path.join(PROJECT_ROOT, "kmeans_model.pkl")
SCALER_PATH      = os.path.join(PROJECT_ROOT, "scaler.pkl")
PRICE_MODEL_PATH = os.path.join(PROJECT_ROOT, "price_predictor.pkl")

# Google Maps mijozini sozlash
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)


def load_db_data():
    """
    PHP‑API orqali kerakli jadvallarni olib,
    narx (announcements), haydovchilar, transport va foydalanuvchilar
    ma'lumotlarini to'rt DataFrame ga qaytaradi.
    """
    # 1) Announcements jadvali → narx ma'lumotlari
    ann = fetch_table("announcements")
    df_price = ann[["pick_up_address", "shipping_address", "price"]].copy()
    # faqat birinchi qism (viloyat)ni ajratamiz
    df_price["raw_from"] = df_price["pick_up_address"].astype(str).map(lambda x: x.split(",")[0])
    df_price["raw_to"]   = df_price["shipping_address"].astype(str).map(lambda x: x.split(",")[0])
    # kirillizatsiya + normalizatsiya
    df_price["from_city"] = df_price["raw_from"].map(
        lambda x: latin_to_cyrillic(x).strip().lower()
    )
    df_price["to_city"] = df_price["raw_to"].map(
        lambda x: latin_to_cyrillic(x).strip().lower()
    )
    # faqat kerakli ustunlarni qoldiramiz
    df_price = df_price[["from_city", "to_city", "price"]]

    # 2) Haydovchilar koordinatalari
    drivers_df = fetch_table("driver_locations")

    # 3) Avtomobillar (og'irlik/haqiqiy hajm)
    my_autos = fetch_table("my_autos")

    # 4) Foydalanuvchilar
    users = fetch_table("users")
    # `id` ustunini `user_id` ga o'zgartirib, keragini ajratamiz
    if "id" in users.columns:
        users = users.rename(columns={"id": "user_id"})
    users = users[["user_id", "fullname", "phone", "status"]]

    return df_price, drivers_df, my_autos, users


def get_coordinates(location: str):
    """
    Google Maps Geocoding API yordamida joylashuvni (lat, lng) qaytaradi.
    """
    try:
        result = gmaps.geocode(location)
        if result:
            loc = result[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None


def load_or_initialize_model():
    """
    Klasterlash modeli va scaler fayllari bor-yo'qligini tekshiradi,
    bo'lmasa dummy data bilan yaratadi.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        dummy = np.array([[1000, 10], [2000, 20]])
        scaler = StandardScaler().fit(dummy)
        model  = MiniBatchKMeans(n_clusters=2, batch_size=2, random_state=42)
        model.partial_fit(scaler.transform(dummy))
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler

model, scaler = load_or_initialize_model()


def online_fit_and_predict(weight: float, volume: float) -> int:
    """
    Yangi weight/volume kiritilganda online klasterlash va bashorat.
    """
    global model, scaler
    X = np.array([[weight, volume]])
    Xs = scaler.transform(X)
    model.partial_fit(Xs)
    joblib.dump(model, MODEL_PATH)
    return int(model.predict(Xs)[0])


def load_or_initialize_price_model():
    """
    SGDRegressor narx bashorat modeli.
    """
    if os.path.exists(PRICE_MODEL_PATH):
        return joblib.load(PRICE_MODEL_PATH)
    pm = SGDRegressor()
    pm.partial_fit([[0, 0]], [0])
    joblib.dump(pm, PRICE_MODEL_PATH)
    return pm


def online_fit_and_predict_price(f_enc: int, t_enc: int, actual_price: float = None) -> int:
    """
    Viloyat kodlari bo'yicha online narx bashorati:
    - actual_price berilsa, modelga partial_fit qilinadi.
    - Najotda 0 dan kichik chiqmasligi uchun max(0, ...) qaytaramiz.
    """
    price_model = load_or_initialize_price_model()
    X = np.array([[f_enc, t_enc]])
    if actual_price is not None:
        price_model.partial_fit(X, [actual_price])
        joblib.dump(price_model, PRICE_MODEL_PATH)
    pred = price_model.predict(X)[0]
    return max(0, int(pred))


def train_price_model_from_db():
    """
    Batch rejimida narx modelini announcements jadvalidan o'qitadi.
    Agar ma'lumot bo'lmasa yoki xato yuz bersa, consolga yozadi.
    """
    try:
        df_price, _, _, _ = load_db_data()
    except Exception as e:
        print(f"❌ Train modeli yuklanmadi, DB xatosi: {e}")
        return

    from_list = df_price["from_city"].tolist()
    to_list   = df_price["to_city"].tolist()
    le = LabelEncoder().fit(from_list + to_list)

    X, y = [], []
    for row in df_price.itertuples(index=False):
        try:
            f_enc = int(le.transform([row.from_city])[0])
            t_enc = int(le.transform([row.to_city])[0])
            p = float(row.price)
            if p > 0:
                X.append([f_enc, t_enc])
                y.append(p)
        except Exception:
            continue

    if not X:
        print("❌ Narx modeli uchun ma'lumot topilmadi.")
        return

    pm = SGDRegressor()
    pm.partial_fit(X, y)
    joblib.dump(pm, PRICE_MODEL_PATH)
    print(f"✅ Narx modeli {len(X)} namunada o‘qitildi.")


def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    """
    Eng mos haydovchilar ro'yxatini tuzadi:
     - Klasterlash bo'yicha,
     - Sig'imi va masofa bo'yicha saralash
    """
    try:
        _, drivers_df, my_autos, users = load_db_data()
    except Exception as e:
        print(f"❌ Haydovchilarni yuklab bo'lmadi, DB xatosi: {e}")
        return []

    # Raqamlarga aylantirish
    drivers_df["latitude"]  = pd.to_numeric(drivers_df["latitude"], errors="coerce")
    drivers_df["longitude"] = pd.to_numeric(drivers_df["longitude"], errors="coerce")
    my_autos["transport_weight"] = pd.to_numeric(my_autos["transport_weight"], errors="coerce")
    my_autos["transport_volume"] = pd.to_numeric(my_autos["transport_volume"], errors="coerce")

    # Jadval biriktirish
    df = (
        drivers_df
        .merge(my_autos, on="user_id")
        .merge(users, on="user_id")
        .dropna(subset=["transport_weight", "transport_volume", "latitude", "longitude"])
        .copy()
    )

    arr = df[["transport_weight", "transport_volume"]].values
    if arr.size == 0:
        return []

    # Lokal scaler + klaster
    scaler_loc = StandardScaler().fit(arr)
    arr_scaled = scaler_loc.transform(arr)
    km = MiniBatchKMeans(
        n_clusters=min(4, len(arr_scaled)),
        batch_size=6,
        random_state=42
    )
    km.partial_fit(arr_scaled)

    # O'z yukingiz klasteri
    inp_scaled = scaler_loc.transform([[weight, volume]])
    cid = int(km.predict(inp_scaled)[0])

    # Saralash: sig'im va masofa bo‘yicha
    df["cluster_id"] = km.predict(arr_scaled)
    same = df[df["cluster_id"] == cid].copy()
    if same.empty:
        return []

    same["capacity_distance"] = np.sqrt(
        (same["transport_weight"] - weight) ** 2 +
        (same["transport_volume"] - volume) ** 2
    )
    same["distance_km"] = np.sqrt(
        (same["latitude"] - lat) ** 2 + (same["longitude"] - lon) ** 2
    ) * 111

    # Eng yaqin 5 ta haydovchini qaytaramiz
    return (
        same
        .sort_values(by=["capacity_distance", "distance_km"])
        .head(5)[
            [
                "fullname", "phone", "transport_model",
                "transport_weight", "transport_volume", "distance_km"
            ]
        ]
        .to_dict(orient="records")
    )
