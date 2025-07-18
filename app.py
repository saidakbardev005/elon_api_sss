import os
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

from config import Config
from services.db import fetch_table
from services.predict_service import (
    get_coordinates,
    online_fit_and_predict,
    online_fit_and_predict_price,
    train_price_model_from_db,
)
from transliteration.latin_to_cyrillic import latin_to_cyrillic

app = Flask(__name__)

# —————— Routes ——————

@app.route('/server')
def index():
    return render_template('index.html')


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    # 1) Parametrlarni olish
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        full_from    = data.get("from", "").strip()
        full_to      = data.get("to", "").strip()
        weight       = data.get("weight")
        volume       = data.get("volume")
        actual_price = data.get("actual_price")
    else:
        full_from    = request.args.get("from", "").strip()
        full_to      = request.args.get("to", "").strip()
        weight       = request.args.get("weight")
        volume       = request.args.get("volume")
        actual_price = request.args.get("actual_price")

    # 2) Bo‘sh parametrlar
    if not full_from or not full_to or weight is None or volume is None:
        return jsonify({
            "error": "Missing required parameters: 'from', 'to', 'weight', 'volume'"
        }), 400

    # 3) Format tekshiruv
    try:
        weight       = float(weight)
        volume       = float(volume)
        actual_price = float(actual_price) if actual_price not in [None, "", "null"] else None
    except (ValueError, TypeError):
        return jsonify({
            "error": "Parameters 'weight', 'volume', and 'actual_price' must be valid numbers"
        }), 400

    # 4) Faqat viloyatlarni ajratish
    frm_region = full_from.split(",")[0].strip()
    to_region  = full_to.split(",")[0].strip()

    # 5) Geolokatsiya
    lat, lon = get_coordinates(frm_region)
    if lat is None or lon is None:
        return jsonify({"error": f"Could not geocode region: {frm_region}"}), 400

    # 6) Bazadagi narx ma’lumotlarini olish
    df_price, _, _, _ = fetch_table("announcements"), None, None, None
    # if you use load_db_data(), ensure it uses fetch_table internally

    vil_from = df_price["from_city"].astype(str).tolist()
    vil_to   = df_price["to_city"].astype(str).tolist()

    # 7) LabelEncoder bilan kodlash
    le       = LabelEncoder().fit(vil_from + vil_to)
    frm_norm = latin_to_cyrillic(frm_region).strip().lower()
    to_norm  = latin_to_cyrillic(to_region).strip().lower()

    try:
        f_enc = int(le.transform([frm_norm])[0])
        t_enc = int(le.transform([to_norm])[0])
    except ValueError:
        return jsonify({
            "error": f"Unknown region names: from='{frm_region}', to='{to_region}'"
        }), 400

    # 8) Narxni bashorat qilish
    price = online_fit_and_predict_price(f_enc, t_enc, actual_price)

    # 9) Klasterlash bashorati
    cluster_id = int(online_fit_and_predict(weight, volume))

    # 10) Eng mos haydovchilarni olish
    drivers = fetch_table("driver_locations")  # or use a dedicated function
    # you may want to call find_best_drivers() here if it's been updated to use fetch_table
    best_drivers = drivers  # replace with find_best_drivers(lat, lon, weight, volume)

    # 11) Javob
    return jsonify({
        "price": price,
        "cluster_id": cluster_id,
        "drivers": best_drivers
    }), 200


@app.route("/api/user/<int:user_id>", methods=["GET"])
def api_get_user(user_id):
    """
    cPanel’dagi PHP API orqali `users` jadvalidan bitta foydalanuvchini oladi.
    """
    try:
        resp = requests.get(
            Config.CPANEL_API_URL,
            params={"id": user_id},
            timeout=5
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"PHP API so‘rovida xato: {e}"}), 502

    data = resp.json()
    if not data:
        return jsonify({"error": "User topilmadi"}), 404

    return jsonify({"user": data[0]}), 200


# —————— Main ——————

if __name__ == "__main__":
    # Deploy va gunicorn ishlatganingizda ham bu kod importda bajariladi:
    print("🚀 Server boshlanyapti...")
    try:
        train_price_model_from_db()
    except Exception as e:
        print(f"[Startup] Narx modelini o‘rganishda xato: {e}")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
