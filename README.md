# Trading Simulator

Django app for confluence / swing strategy, stage analysis, scanners, paper trading, and backtests (NSE-focused).

## Local setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

### One-click (Windows)

1. Double-click **`Trading Simulator`** on your Desktop  
   (or run `start_app.bat` in this folder)
2. A console window starts the server and your browser opens the app  
3. Leave the console open while using the app; close it to stop the server  

If the desktop icon is missing:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_desktop_shortcut.ps1
```

## Deploy (always online)

See **[DEPLOY_PYTHONANYWHERE.md](DEPLOY_PYTHONANYWHERE.md)** for a full PythonAnywhere guide  
(clone → venv → WSGI → migrate → daily price task → live URL).

## Stack

- Django 5, SQLite  
- pandas / numpy / scikit-learn / plotly  
- yfinance for NSE OHLCV sync  
