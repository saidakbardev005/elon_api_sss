# services/external_api.py

import requests

CPANEL_API_URL = "https://u-logistic-ai.uz/api.php"  # o‘zingizniki bilan almashtiring

def get_user_by_id(user_id: int):
    """
    cPanel’dagi PHP API orqali `users` jadvalidan user ma'lumotini oladi.
    """
    try:
        resp = requests.get(
            CPANEL_API_URL,
            params={"id": user_id},
            timeout=5
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        # xatoni ko‘tarish yoki loglash mumkin
        raise RuntimeError(f"PHP API so‘rovida xato: {e}")

    # JSON sifatida qaytariladi (ro‘yxat ichida bitta yoki bo‘sh ro‘yxat)
    return resp.json()
