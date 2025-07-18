# services/db.py

import requests
import pandas as pd
from config import Config

def fetch_table(table_name: str) -> pd.DataFrame:
    """
    cPanel’dagi PHP‐API orqali berilgan jadvalni oladi
    va pandas DataFrame ga aylantirib qaytaradi.
    """
    try:
        resp = requests.get(
            Config.CPANEL_API_URL,
            params={"table": table_name},
            timeout=10
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Jadvalni olishda xato (table={table_name}): {e}")

    data = resp.json()
    # Agar bo'sh ro'yxat qaytsa, bo'sh DataFrame qaytariladi
    return pd.DataFrame(data)

def fetch_user(user_id: int) -> dict:
    """
    cPanel’dagi PHP‐API orqali `users` jadvalidan bitta foydalanuvchini oladi.
    Agar topilmasa ValueError ko‘taradi.
    """
    try:
        resp = requests.get(
            Config.CPANEL_API_URL,
            params={"id": user_id},
            timeout=5
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Userni olishda xato (id={user_id}): {e}")

    data = resp.json()
    if not data:
        raise ValueError(f"User topilmadi (id={user_id})")
    return data[0]
