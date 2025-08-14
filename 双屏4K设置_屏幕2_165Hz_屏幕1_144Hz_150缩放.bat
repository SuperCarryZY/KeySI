@echo off
chcp 65001 >nul
title 双屏4K设置 - 屏幕2(165Hz) + 屏幕1(144Hz) + 150%缩放
color 0B

echo.
echo ================================================================
echo    🖥️  双屏4K设置 - 屏幕2(165Hz) + 屏幕1(144Hz) + 150%%缩放
echo ================================================================
echo.
echo 📋 即将执行的操作：
echo    🔹 启用双屏扩展模式
echo    🔹 屏幕2: 4K (3840x2160) + 165Hz + 150%%缩放
echo    🔹 屏幕1: 4K (3840x2160) + 144Hz + 150%%缩放  
echo    🔹 设置系统缩放：150%%
echo.
echo ⏳ 开始执行设置...
echo.

:: 步骤1：启用双屏扩展模式
echo 🖥️  [1/4] 启用双屏扩展模式...
displayswitch.exe /extend
if %errorlevel%==0 (
    echo    ✅ 双屏扩展模式已启用
    timeout /t 3 /nobreak >nul
) else (
    echo    ❌ 双屏切换失败
    echo    💡 请手动按 Win+P 选择"扩展"
)

:: 步骤2：设置150%缩放 (144 DPI)
echo 🔍 [2/4] 设置缩放比例为150%%...
reg add "HKCU\Control Panel\Desktop" /v LogPixels /t REG_DWORD /d 144 /f >nul 2>&1
reg add "HKCU\Control Panel\Desktop" /v Win8DpiScaling /t REG_DWORD /d 1 /f >nul 2>&1

if %errorlevel%==0 (
    echo    ✅ 150%%缩放设置成功 ^(需要重新登录生效^)
) else (
    echo    ❌ 缩放设置失败
)

:: 步骤3：设置主屏幕4K分辨率
echo 📐 [3/4] 设置主屏幕4K分辨率...
powershell -Command "& {try {Add-Type -AssemblyName System.Windows.Forms; $screen = [System.Windows.Forms.Screen]::PrimaryScreen; $currentW = $screen.Bounds.Width; $currentH = $screen.Bounds.Height; Write-Host \"主屏幕当前分辨率: ${currentW}x${currentH}\"; if ($currentW -eq 3840 -and $currentH -eq 2160) {Write-Host '✅ 主屏幕已经是4K分辨率'} else {Write-Host '⚠️  主屏幕需要手动设置为4K'}} catch {Write-Host '⚠️  请手动检查主屏幕分辨率'}}"

:: 步骤4：尝试使用PowerShell设置分辨率和刷新率
echo 🔄 [4/4] 尝试设置刷新率和分辨率...
powershell -Command "& {Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class DisplayAPI { [DllImport(\"user32.dll\")] public static extern int ChangeDisplaySettings(IntPtr devMode, int flags); [DllImport(\"user32.dll\")] public static extern bool EnumDisplaySettings(string deviceName, int modeNum, ref DEVMODE devMode); [DllImport(\"user32.dll\")] public static extern bool EnumDisplayDevices(IntPtr lpDevice, int iDevNum, ref DISPLAY_DEVICE lpDisplayDevice, int dwFlags); [StructLayout(LayoutKind.Sequential)] public struct DEVMODE { [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName; public short dmSpecVersion; public short dmDriverVersion; public short dmSize; public short dmDriverExtra; public int dmFields; public int dmPositionX; public int dmPositionY; public int dmDisplayOrientation; public int dmDisplayFixedOutput; public short dmColor; public short dmDuplex; public short dmYResolution; public short dmTTOption; public short dmCollate; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName; public short dmLogPixels; public int dmBitsPerPel; public int dmPelsWidth; public int dmPelsHeight; public int dmDisplayFlags; public int dmDisplayFrequency; public int dmICMMethod; public int dmICMIntent; public int dmMediaType; public int dmDitherType; public int dmReserved1; public int dmReserved2; public int dmPanningWidth; public int dmPanningHeight; } [StructLayout(LayoutKind.Sequential)] public struct DISPLAY_DEVICE { public int cb; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString; public int StateFlags; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey; } }'; try { Write-Host '🔄 尝试设置主屏幕为4K@165Hz...'; $devMode = New-Object DisplayAPI+DEVMODE; $devMode.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($devMode); if([DisplayAPI]::EnumDisplaySettings($null, -1, [ref]$devMode)) { Write-Host \"主屏幕当前: $($devMode.dmPelsWidth)x$($devMode.dmPelsHeight)@$($devMode.dmDisplayFrequency)Hz\"; $devMode.dmPelsWidth = 3840; $devMode.dmPelsHeight = 2160; $devMode.dmDisplayFrequency = 165; $devMode.dmFields = 0x580000; $result = [DisplayAPI]::ChangeDisplaySettings([ref]$devMode, 1); if($result -eq 0) { Write-Host '✅ 主屏幕4K@165Hz设置成功' } else { Write-Host '⚠️  主屏幕自动设置失败' } } } catch { Write-Host '⚠️  需要手动设置分辨率和刷新率' }}"

timeout /t 2 /nobreak >nul

:: 验证设置
echo.
echo 📊 [验证] 检查当前设置...
powershell -Command "& {try {Add-Type -AssemblyName System.Windows.Forms; $screens = [System.Windows.Forms.Screen]::AllScreens; Write-Host \"检测到 $($screens.Count) 个显示器:\"; for($i=0; $i -lt $screens.Count; $i++) { $screen = $screens[$i]; Write-Host \"显示器 $($i+1): $($screen.Bounds.Width) x $($screen.Bounds.Height)\" } } catch {Write-Host '无法获取显示器信息'}}"

:: 显示结果和手动调整指导
echo.
echo ================================================================
echo 📊 设置完成！
echo ================================================================
echo.
echo 📝 已尝试设置：
echo    🖥️  显示模式：双屏扩展
echo    📐 目标分辨率：两个屏幕都为4K ^(3840x2160^)
echo    🔄 目标刷新率：屏幕2为165Hz，屏幕1为144Hz
echo    🔍 缩放比例：150%% ^(需要重新登录生效^)
echo.
echo 💡 重要提示：
echo    • 150%%缩放需要重新登录Windows才能生效
echo    • 部分设置可能需要手动调整
echo.
echo 🔧 手动调整步骤：
echo    1. 右键桌面 → 显示设置
echo    2. 选择"显示器1"：
echo       • 分辨率 → 3840 x 2160
echo       • 高级显示设置 → 刷新率 → 144Hz
echo    3. 选择"显示器2"：
echo       • 分辨率 → 3840 x 2160  
echo       • 高级显示设置 → 刷新率 → 165Hz
echo    4. 缩放和布局 → 150%%
echo    5. 保存设置后重新登录
echo.
echo 🎯 快捷操作：
echo    • Win+P: 投影设置
echo    • Win+I: 打开设置
echo    • 搜索"显示"快速找到显示设置
echo.
echo ⚠️  是否需要立即注销以应用150%%缩放？
echo.

set /p logout_choice=输入 Y 立即注销，任意键跳过: 

if /i "%logout_choice%"=="Y" (
    echo.
    echo 🔄 保存所有工作后，5秒后自动注销...
    timeout /t 5
    shutdown /l
) else (
    echo.
    echo 👍 已跳过自动注销
    echo 💡 请记得稍后手动注销以应用150%%缩放设置
)

echo.
echo ================================================================

echo.
echo 按任意键退出...
pause >nul
