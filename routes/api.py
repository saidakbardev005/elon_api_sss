# flask_api_copy/flask_api/routes/api.py

from flask import Blueprint, request, jsonify
from transliteration.latin_to_cyrillic import latin_to_cyrillic
from services.predict_service import (
    get_coordinates,
    predict_price,
    find_best_drivers,
    load_csv_data
)
from sklearn.preprocessing import LabelEncoder

api = Blueprint("api", __name__)

def is_cyrillic(text: str) -> bool:
    for ch in text:
        if ('А' <= ch <= 'я') or ch in ('Ёё'):
            return True
    return False

@api.route("/predict", methods=["GET", "POST"])
def predict_route():
    # 1) Parametrlarni olish
    if request.method == "POST":
        data     = request.get_json() or {}
        raw_from = data.get("from")
        raw_to   = data.get("to")
        weight   = data.get("weight")
        volume   = data.get("volume")
    else:
        raw_from = request.args.get("from")
        raw_to   = request.args.get("to")
        weight   = request.args.get("weight")
        volume   = request.args.get("volume")

    # 2) Majburiy parametrlar
    if not raw_from or not raw_to or weight is None or volume is None:
        return jsonify({
            "error": "Missing required parameters: 'from', 'to', 'weight', 'volume'"
        }), 400

    # 3) weight/volume float
    try:
        weight = float(weight)
        volume = float(volume)
    except (ValueError, TypeError):
        return jsonify({"error": "Parameters 'weight' and 'volume' must be numbers"}), 400

    # 4) “region,district” → faqat region
    frm_region = raw_from.split(",", 1)[0].strip()
    to_region  = raw_to.split(",", 1)[0].strip()

    # 5) Kiruvchi nomni lowercase kirillga normallashtirish
    if is_cyrillic(frm_region):
        f_norm = frm_region
    else:
        f_norm = latin_to_cyrillic(frm_region)
    f_norm = f_norm.strip().lower()

    if is_cyrillic(to_region):
        t_norm = to_region
    else:
        t_norm = latin_to_cyrillic(to_region)
    t_norm = t_norm.strip().lower()

    # 6) DB’dan narx jadvali
    try:
        df_price, _, _, _ = load_csv_data()
    except Exception as e:
        return jsonify({"error": f"Failed loading price data: {e}"}), 500

    # 7) DB’dagi qiymatlarni ham normalize qilamiz
    df_price["from_norm"] = (
        df_price["from_city"]
        .fillna("")
        .astype(str)
        .apply(lambda x: latin_to_cyrillic(x).strip().lower())
    )
    df_price["to_norm"] = (
        df_price["to_city"]
        .fillna("")
        .astype(str)
        .apply(lambda x: latin_to_cyrillic(x).strip().lower())
    )

    # 8) LabelEncoder faqat normallashtirilgan ro‘yxatga fit qilamiz
    le = LabelEncoder().fit(
        list(df_price["from_norm"]) +
        list(df_price["to_norm"])
    )

    # 9) Kodlash va mavjudligini tekshirish
    if f_norm not in le.classes_ or t_norm not in le.classes_:
        return jsonify({
            "error": f"Unknown region names: from={frm_region}, to={to_region}"
        }), 400

    f_enc = int(le.transform([f_norm])[0])
    t_enc = int(le.transform([t_norm])[0])

    # 10) Narxni bashorat qilamiz
    try:
        predicted_price = predict_price(f_enc, t_enc)
    except Exception as e:
        return jsonify({"error": f"Price prediction failed: {e}"}), 500

    # 11) Geokodlash (region bo‘yicha)
    lat, lon = get_coordinates(frm_region)
    if lat is None or lon is None:
        return jsonify({"error": f"Unable to geocode region: {frm_region}"}), 400

    # 12) Haydovchilarni tanlaymiz
    try:
        best_drivers = find_best_drivers(lat, lon, weight, volume)
    except Exception as e:
        return jsonify({"error": f"Driver selection failed: {e}"}), 500

    # 13) Javobni qaytaramiz
    return jsonify({
        "predicted_price": predicted_price,
        "drivers": best_drivers
    }), 200
