# Upload KeySI project to GitHub
Write-Host "========================================" -ForegroundColor Green
Write-Host "Uploading KeySI project to GitHub" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Then run this script again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if this is already a git repository
if (Test-Path ".git") {
    Write-Host "Repository already initialized" -ForegroundColor Yellow
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
    git branch -M main
}

# Check if remote exists
try {
    $remoteUrl = git remote get-url origin 2>$null
    Write-Host "Remote origin already exists: $remoteUrl" -ForegroundColor Yellow
} catch {
    Write-Host "Adding remote origin..." -ForegroundColor Cyan
    git remote add origin git@github.com:SuperCarryZY/KeySI.git
}

# Add all files
Write-Host "Adding files to git..." -ForegroundColor Cyan
git add .

# Show status
Write-Host "Current git status:" -ForegroundColor Cyan
git status --short

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Cyan
git commit -m "Update finaluimodified.py - Complete Chinese to English translation

- Translated all Chinese text to English including:
  * Function docstrings and comments
  * UI text and button labels  
  * Debug messages and error messages
  * Technical terminology
- Fixed debug_info variable initialization bug
- Maintained all original functionality
- All interface text now in English"

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push -u origin main

Write-Host "========================================" -ForegroundColor Green
Write-Host "Upload completed successfully!" -ForegroundColor Green
Write-Host "Your repository: https://github.com/SuperCarryZY/KeySI" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Read-Host "Press Enter to exit"

