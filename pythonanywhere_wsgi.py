"""
Copy this into your PythonAnywhere WSGI file (Web tab → WSGI configuration file),
or paste the body into /var/www/<username>_pythonanywhere_com_wsgi.py

Replace YOUR_USERNAME with your PythonAnywhere username.
"""
import os
import sys

# --- Project path ---
project_home = "/home/YOUR_USERNAME/Trading_Simulator"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# --- Production environment (edit values) ---
os.environ["DJANGO_SETTINGS_MODULE"] = "confluence_trader.settings"
os.environ.setdefault("DJANGO_DEBUG", "0")
os.environ.setdefault(
    "DJANGO_SECRET_KEY",
    "REPLACE_WITH_A_LONG_RANDOM_SECRET",
)
os.environ.setdefault(
    "DJANGO_ALLOWED_HOSTS",
    "YOUR_USERNAME.pythonanywhere.com",
)
os.environ.setdefault(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    "https://YOUR_USERNAME.pythonanywhere.com",
)
# Prefer scheduled task for price sync (Tasks tab), not web-worker threads
os.environ.setdefault("NSE_AUTO_SYNC_ENABLED", "0")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
