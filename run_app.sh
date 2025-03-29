#!/bin/bash

# เข้าสู่โฟลเดอร์โปรเจค
cd "$(dirname "$0")"

# เปิดใช้งาน virtual environment
if [ -d "new_env" ]; then
    source new_env/bin/activate
else
    echo "Virtual environment not found. Creating..."
    python3.9 -m venv new_env
    source new_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# รัน Streamlit app ด้วย Python จาก virtual environment
python -m streamlit run mt5_chat_app.py "$@" 