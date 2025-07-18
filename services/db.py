import requests
import pandas as pd
from config import Config

def fetch_table(table_name: str) -> pd.DataFrame:
    """
    cPanel’dagi PHP‑API orqali berilgan jadvalni oladi
    va pandas DataFrame ga aylantirib qaytaradi.
    """
    url     = Config.CPANEL_API_URL
    params  = {"table": table_name}
    headers = {
        "Accept":     "application/json",
        "User-Agent": "python-requests"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    # API xatosini aniqlash
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"API xatosi (table={table_name}): {data['error']}")

    # Bitta dict qaytganda uni list ga o‘rash
    if isinstance(data, dict):
        data = [data]

    # Oxirgi qadam: list-of-dicts → DataFrame
    return pd.DataFrame(data)


def fetch_user(user_id: int) -> dict:
    """
    cPanel’dagi PHP‑API orqali bitta user oladi.
    Agar topilmasa ValueError, API xatosi bo‘lsa RuntimeError.
    """
    url     = Config.CPANEL_API_URL
    params  = {"id": user_id}
    headers = {
        "Accept":     "application/json",
        "User-Agent": "python-requests"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=5)
    resp.raise_for_status()

    data = resp.json()
    # API xatosi?
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"API xatosi (id={user_id}): {data['error']}")

    # List bo‘lsa, birinchi element
    if isinstance(data, list) and data:
        return data[0]

    # Hech nima bo‘lmasa
    raise ValueError(f"User topilmadi (id={user_id})")
