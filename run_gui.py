"""Script khởi động nhanh giao diện Desktop GUI từ thư mục gốc."""

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc được ưu tiên
PROJECT_ROOT = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "MODULE")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GUI.gui_app import main

if __name__ == "__main__":
    main()

