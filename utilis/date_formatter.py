import winreg
from datetime import datetime, timezone

def get_windows_datetime_format() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\International") as key:
            short_date = winreg.QueryValueEx(key, "sShortDate")[0]
            time_format = winreg.QueryValueEx(key, "sTimeFormat")[0]
            
            # Convert Windows date tokens to Python strftime tokens
            # Longest patterns first to prevent substring matching issues
            date_map = [
                ("yyyy", "%Y"),
                ("yy", "%y"),
                ("MMMM", "%B"),
                ("MMM", "%b"),
                ("MM", "%m"),
                ("M", "%m"),
                ("dd", "%d"),
                ("d", "%d"),
            ]
            py_date = short_date
            for win_tok, py_tok in date_map:
                py_date = py_date.replace(win_tok, py_tok)
                
            # Convert Windows time tokens to Python strftime tokens
            time_map = [
                ("HH", "%H"),
                ("H", "%H"),
                ("hh", "%I"),
                ("h", "%I"),
                ("mm", "%M"),
                ("m", "%M"),
                ("ss", "%S"),
                ("s", "%S"),
                ("tt", "%p"),
                ("t", "%p"),
            ]
            py_time = time_format
            for win_tok, py_tok in time_map:
                py_time = py_time.replace(win_tok, py_tok)
                
            return f"{py_date} {py_time}"
    except Exception:
        # Fallback to a standard readable format
        return "%d-%b-%Y %I:%M:%S %p"

def format_local_datetime(utc_iso_str: str) -> str:
    if not utc_iso_str:
        return ""
    try:
        if utc_iso_str.endswith("Z"):
            utc_iso_str = utc_iso_str[:-1] + "+00:00"
        # Handle formats like "2026-07-23 03:39:30" (replace spaces with T)
        if " " in utc_iso_str and "T" not in utc_iso_str:
            parts = utc_iso_str.split(" ")
            utc_iso_str = f"{parts[0]}T{parts[1]}"
        dt = datetime.fromisoformat(utc_iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        from datetime import timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        local_dt = dt.astimezone(ist_tz)
        return local_dt.strftime("%d-%b-%Y %I:%M %p IST")
    except Exception:
        return utc_iso_str
