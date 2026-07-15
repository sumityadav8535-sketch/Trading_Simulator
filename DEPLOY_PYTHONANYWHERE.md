# Deploy Trading Simulator on PythonAnywhere

Follow these steps once. After that, open your site in a browser — no local `runserver`.

**Repo:** https://github.com/sumityadav8535-sketch/Trading_Simulator  
**Time:** about 20–40 minutes  
**Account:** free “Beginner” works for personal use (CPU/seconds limits apply)

---

## 1. Create a PythonAnywhere account

1. Go to [https://www.pythonanywhere.com](https://www.pythonanywhere.com) and sign up.
2. Note your **username** (your site will be `https://USERNAME.pythonanywhere.com`).

---

## 2. Clone the project (Bash console)

Open **Consoles → Bash** and run (replace `YOUR_USERNAME`):

```bash
cd ~
git clone https://github.com/sumityadav8535-sketch/Trading_Simulator.git
cd Trading_Simulator
```

If the repo is private, use a [GitHub personal access token](https://github.com/settings/tokens) as the password when Git asks.

---

## 3. Create a virtualenv and install packages

PythonAnywhere free accounts often have Python **3.10** or **3.11**. Check with `python3.10 --version` (or try `3.11` / `3.12`).

```bash
cd ~/Trading_Simulator
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pandas` / `scikit-learn` fail on free tier, try:

```bash
pip install "numpy<2" "pandas>=2.2.0" "scikit-learn>=1.5.0"
pip install -r requirements.txt
```

---

## 4. Create a secret key

In the same Bash console:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

Copy the output — you will paste it into the WSGI file.

---

## 5. Configure the web app

1. Open **Web** tab → **Add a new web app**.
2. Choose **Manual configuration** (not “Django” wizard — we already have a project).
3. Select the same Python version as your venv (e.g. 3.10).
4. Under **Virtualenv**, set:

   ```text
   /home/YOUR_USERNAME/Trading_Simulator/venv
   ```

5. Open **WSGI configuration file** (link near the top of the Web tab).
6. **Delete** the default Flask/Django demo content.
7. Paste the contents of `pythonanywhere_wsgi.py` from this repo, then fix:

   | Placeholder | Replace with |
   |-------------|--------------|
   | `YOUR_USERNAME` | your PA username |
   | `REPLACE_WITH_A_LONG_RANDOM_SECRET` | the secret from step 4 |

   Example for user `sumit`:

   ```python
   project_home = "/home/sumit/Trading_Simulator"
   os.environ["DJANGO_ALLOWED_HOSTS"] = "sumit.pythonanywhere.com"
   os.environ["DJANGO_CSRF_TRUSTED_ORIGINS"] = "https://sumit.pythonanywhere.com"
   ```

8. **Save** the WSGI file.
9. Click the green **Reload** button on the Web tab.

---

## 6. Database, admin user, static files

In **Bash** (venv activated):

```bash
cd ~/Trading_Simulator
source venv/bin/activate

export DJANGO_DEBUG=0
export DJANGO_SECRET_KEY='paste-same-secret-as-wsgi'
export DJANGO_ALLOWED_HOSTS='YOUR_USERNAME.pythonanywhere.com'
export DJANGO_CSRF_TRUSTED_ORIGINS='https://YOUR_USERNAME.pythonanywhere.com'
export NSE_AUTO_SYNC_ENABLED=0

python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

### Load market data (important)

GitHub does **not** include `db.sqlite3` or large price caches. Load data on the server:

**Option A — from yfinance (slow first time, needs outbound internet):**

```bash
# If you have management commands for universe + prices:
python manage.py load_nse_data   # if this command exists / fits your workflow
python manage.py update_nse_prices
python manage.py mark_nifty100   # optional helpers if needed
python manage.py mark_nifty200
python manage.py load_nifty50_index
```

**Option B — upload your local DB (faster):**

1. On your PC, zip `db.sqlite3` (and optionally `data/nse_data.sqlite`).
2. PythonAnywhere → **Files** → upload into `/home/YOUR_USERNAME/Trading_Simulator/`.
3. Or use the PA “Upload a file” UI.

Then **Reload** the web app again.

---

## 7. Schedule daily price updates (Tasks tab)

Web workers should **not** run long yfinance syncs. Use a scheduled task:

1. Open **Tasks**.
2. Add a **Scheduled task** (daily, e.g. **11:00 UTC** ≈ after NSE close + buffer, or evening IST).
3. Command:

```bash
/home/YOUR_USERNAME/Trading_Simulator/venv/bin/python /home/YOUR_USERNAME/Trading_Simulator/manage.py update_nse_prices
```

If free accounts only allow one always-on task or limited schedules, run price sync manually from Bash when needed:

```bash
source ~/Trading_Simulator/venv/bin/activate
cd ~/Trading_Simulator
python manage.py update_nse_prices
```

---

## 8. Open your live site

Visit:

```text
https://YOUR_USERNAME.pythonanywhere.com
```

Admin:

```text
https://YOUR_USERNAME.pythonanywhere.com/admin/
```

---

## Updating the app later

```bash
cd ~/Trading_Simulator
source venv/bin/activate
git pull
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
```

Then **Web → Reload**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **DisallowedHost** | Set `DJANGO_ALLOWED_HOSTS` in WSGI to exact hostname |
| **CSRF verification failed** | Set `DJANGO_CSRF_TRUSTED_ORIGINS` to `https://USERNAME.pythonanywhere.com` |
| **Static CSS/JS missing** | Run `collectstatic`, confirm WhiteNoise in settings, Reload |
| **Error log** | Web tab → **Error log** / **Server log** |
| **ImportError / ModuleNotFound** | Virtualenv path wrong, or `pip install -r requirements.txt` inside venv |
| **Site works but no stocks/prices** | Upload DB or run load/update management commands |
| **CPU seconds exhausted (free)** | Heavy backtests hit free limits — run less often or upgrade Hacker plan |
| **yfinance blocked / timeouts** | Free tier outbound can be flaky; upload local data or upgrade |

---

## Security checklist (do this)

- [ ] `DJANGO_DEBUG=0` in WSGI  
- [ ] Strong unique `DJANGO_SECRET_KEY`  
- [ ] `createsuperuser` done; use a strong password  
- [ ] Do not commit real secrets to GitHub  
- [ ] Optional: protect views with login if you only want yourself to use the app  

---

## Local development (unchanged)

On your PC, defaults stay developer-friendly:

```bash
# no env vars needed
python manage.py runserver
```

`DEBUG` defaults to on, auto price-sync stays enabled locally.
