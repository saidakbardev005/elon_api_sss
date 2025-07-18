# services/db.py

import requests
import pandas as pd
from config import Config

def fetch_table(table_name: str) -> pd.DataFrame:
    """
    cPanel’dagi PHP‑API orqali berilgan jadvalni oladi
    va pandas DataFrame ga aylantirib qaytaradi.
    """
    url = Config.CPANEL_API_URL
    params = {"table": table_name}
    headers = {
        "Accept": "application/json",
        "User-Agent": "python-requests"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Jadvalni olishda xato (table={table_name}): {e}")

    data = resp.json()
    # Agar API error obyekt qaytarilsa, xatoni ko‘taramiz
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"API xatosi (table={table_name}): {data['error']}")

    # Bo‘sh ro‘yxat ham bo‘lishi mumkin: shunda bo‘sh DataFrame qaytadi
    return pd.DataFrame(data)


def fetch_user(user_id: int) -> dict:
    """
    cPanel’dagi PHP‑API orqali `users` jadvalidan bitta foydalanuvchini oladi.
    Agar topilmasa ValueError ko‘taradi.
    """
    url = Config.CPANEL_API_URL
    params = {"id": user_id}
    headers = {
        "Accept": "application/json",
        "User-Agent": "python-requests"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Userni olishda xato (id={user_id}): {e}")

    data = resp.json()
    # Agar list bo‘lsa, birinchi yozuvni qaytaramiz
    if isinstance(data, list) and data:
        return data[0]
    # Agar bo‘sh yoki xato obyekt bo‘lsa, error ko‘tamiz
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"API xatosi (id={user_id}): {data['error']}")
    raise ValueError(f"User topilmadi (id={user_id})")
