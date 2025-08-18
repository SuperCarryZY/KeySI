@echo off
chcp 65001 >nul
title 检查缩放状态
color 0A

echo.
echo ================================
echo    🔍 检查当前缩放状态
echo ================================
echo.

:: 检查注册表中的DPI设置
echo 📋 检查注册表DPI设置...
for /f "tokens=3" %%i in ('reg query "HKCU\Control Panel\Desktop" /v LogPixels 2^>nul ^| findstr LogPixels') do set RegDPI=%%i

if "%RegDPI%"=="" (
    echo    ⚠️  注册表中未找到DPI设置 ^(使用系统默认^)
    set RegDPI=96
) else (
    echo    📊 注册表DPI值: %RegDPI%
)

:: 计算缩放比例
if "%RegDPI%"=="96" (
    set Scale=100%%
    set ScaleNum=100
) else if "%RegDPI%"=="120" (
    set Scale=125%%
    set ScaleNum=125
) else if "%RegDPI%"=="144" (
    set Scale=150%%
    set ScaleNum=150
) else (
    set /a ScaleNum=%RegDPI%*100/96
    set Scale=%ScaleNum%%%
)

echo    🔍 计算缩放比例: %Scale%

:: 检查系统当前DPI
echo.
echo 📊 检查系统当前DPI...
powershell -Command "& {try {Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::SetHighDpiMode('SystemAware'); $graphics = [System.Drawing.Graphics]::FromHwnd([System.IntPtr]::Zero); $dpi = $graphics.DpiX; $scale = [math]::Round($dpi / 96 * 100); Write-Host \"系统DPI: $dpi (缩放: $scale%%)\"; $graphics.Dispose()} catch {Write-Host '无法获取系统DPI'}}"

echo.
echo ================================
echo 📋 分析结果：
echo ================================

if "%ScaleNum%"=="100" (
    echo ✅ 当前设置为100%%缩放 ^(默认^)
    echo 💡 100%%缩放通常立即生效，无需注销
    echo.
    echo 🔄 如果程序显示异常：
    echo    • 重启相关程序即可
    echo    • 无需注销整个系统
) else (
    echo ⚠️  当前设置为%Scale%缩放 ^(非默认^)
    echo 💡 非默认缩放建议注销后生效
    echo.
    echo 🔄 建议操作：
    echo    • 保存所有工作
    echo    • 注销并重新登录
    echo    • 或者重启计算机
)

echo.
echo 🎯 快速注销命令：
echo    shutdown /l
echo.
echo ================================

echo.
echo 是否需要立即注销? ^(Y/N^)
set /p choice=请选择: 

if /i "%choice%"=="Y" (
    echo.
    echo 🔄 3秒后自动注销...
    timeout /t 3 /nobreak
    shutdown /l
) else (
    echo.
    echo 👍 已取消注销
)

echo.
pause

