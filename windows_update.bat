@echo off
setlocal

echo Updating FieldWeave...

echo Fetching latest changes from git...
git fetch origin main
if errorlevel 1 (
    echo ERROR: git fetch failed. Check your connection or repository status.
    pause
    exit /b 1
)

echo Switching to the main branch...
git checkout -B main origin/main
if errorlevel 1 (
    echo ERROR: git checkout failed. Check for local changes that would be overwritten.
    pause
    exit /b 1
)

if not exist venv\ (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo FieldWeave updated successfully.
pause
endlocal
