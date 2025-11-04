@echo off
echo Starting WormGPT System...

REM Start model runner
start wsl -e bash -c "cd '/mnt/c/Users/one by one/Downloads/New folder/wormgpt_-the-obsidian-flame' && source venv/bin/activate && python run.py"

REM Start API server
start wsl -e bash -c "cd '/mnt/c/Users/one by one/Downloads/New folder/wormgpt_-the-obsidian-flame' && source venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8000 --reload"

REM Start frontend
start cmd /k "cd \"C:\Users\one by one\Downloads\New folder\wormgpt_-the-obsidian-flame\" && npm run dev"

echo ✅ All systems launched!
pause
