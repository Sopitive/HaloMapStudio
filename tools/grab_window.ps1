# grab_window.ps1 — capture a running app's window to a PNG for ground-truth comparison.
# Usage:  powershell -File grab_window.ps1 -Proc sapien -Out C:\path\shot.png
# Captures the CLIENT area of the first matching process's main window via the desktop
# composite (works for D3D/GPU windows as long as the window is visible/unobscured).
param(
  [string]$Proc = "sapien",
  [string]$Out  = "sapien_grab.png"
)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WGrab {
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  public struct RECT { public int L,T,R,B; }
  public struct POINT { public int X,Y; }
}
"@
$p = Get-Process $Proc -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $p) { Write-Output "NOT FOUND: process '$Proc' with a window"; exit 1 }
$h = $p.MainWindowHandle
[WGrab]::ShowWindow($h, 5) | Out-Null            # SW_SHOW
[WGrab]::SetForegroundWindow($h) | Out-Null      # best-effort (may be blocked by focus lock)
Start-Sleep -Milliseconds 500
$r = New-Object WGrab+RECT; [WGrab]::GetClientRect($h, [ref]$r) | Out-Null
$tl = New-Object WGrab+POINT; $tl.X = 0; $tl.Y = 0; [WGrab]::ClientToScreen($h, [ref]$tl) | Out-Null
$w = $r.R - $r.L; $ht = $r.B - $r.T
if ($w -le 0 -or $ht -le 0) { Write-Output "bad client rect ${w}x${ht}"; exit 1 }
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap $w, $ht
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($tl.X, $tl.Y, 0, 0, (New-Object System.Drawing.Size $w, $ht))
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output ("saved {0} ({1}x{2}) title='{3}'" -f $Out, $w, $ht, $p.MainWindowTitle)
