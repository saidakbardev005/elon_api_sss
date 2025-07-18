import os

class Config:
    # cPanel’dagi PHP‑API endpoint URL’i
    CPANEL_API_URL = os.getenv(
        "CPANEL_API_URL",
        "https://u-logistic-ai.uz/api.php"
    )

    # Google Maps API kaliti
    GOOGLE_MAPS_API_KEY = os.getenv(
        "GOOGLE_MAPS_API_KEY",
        "AIzaSyAd4rEAQqf58fCJGABqW99teDP9BcuyN08"
    )
