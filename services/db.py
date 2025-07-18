# services/db.py

import requests
import pandas as pd
from config import Config

def fetch_table(table_name: str) -> pd.DataFrame:
    """
    cPanel’dagi PHP‐API orqali berilgan jadvalni POST so‘rovi bilan oladi
    va pandas DataFrame ga aylantirib qaytaradi.
    """
    url = Config.CPANEL_API_URL
    payload = {"table": table_name}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Jadvalni olishda xato (table={table_name}): {e}")

    data = resp.json()
    return pd.DataFrame(data)


def fetch_user(user_id: int) -> dict:
    """
    cPanel’dagi PHP‐API orqali `users` jadvalidan bitta foydalanuvchini oladi.
    """
    url = Config.CPANEL_API_URL
    payload = {"id": user_id}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Userni olishda xato (id={user_id}): {e}")

    data = resp.json()
    if not data:
        raise ValueError(f"User topilmadi (id={user_id})")
    return data[0]
