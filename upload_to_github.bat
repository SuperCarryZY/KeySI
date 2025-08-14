@echo off
echo ========================================
echo Uploading KeySI project to GitHub
echo ========================================

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    echo Then run this script again
    pause
    exit /b 1
)

REM Check if this is already a git repository
if exist ".git" (
    echo Repository already initialized
) else (
    echo Initializing git repository...
    git init
    git branch -M main
)

REM Add remote if not exists
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo Adding remote origin...
    git remote add origin git@github.com:SuperCarryZY/KeySI.git
) else (
    echo Remote origin already exists
)

REM Add all files
echo Adding files to git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Update finaluimodified.py - Complete Chinese to English translation

- Translated all Chinese text to English including:
  * Function docstrings and comments
  * UI text and button labels  
  * Debug messages and error messages
  * Technical terminology
- Fixed debug_info variable initialization bug
- Maintained all original functionality
- All interface text now in English"

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main

echo ========================================
echo Upload completed successfully!
echo Your repository: https://github.com/SuperCarryZY/KeySI
echo ========================================
pause

