"""AutoPath: dự đoán đường đi (ego-path) cho xe tự lái."""
import sys

# Console Windows mặc định cp1252 không in được tiếng Việt khi chuyển hướng ra file.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
