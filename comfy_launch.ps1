# Launch ComfyUI from the recorded command line in .comfy_launch.json.
#
# The cache entry can have cwd=null (the producer records argv only when it
# cannot see the live process), and the relative "ComfyUI\main.py" argument
# then needs a working directory or the server exits rc=2 and :8188 never
# comes back. The Python side already recovers it (_recall_comfy_launch);
# this is the same walk for the watchdog.
#
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File comfy_launch.ps1
$ErrorActionPreference = 'Stop'
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$cache = Join-Path $proj '.comfy_launch.json'
if (-not (Test-Path -LiteralPath $cache)) { Write-Output 'no .comfy_launch.json'; exit 1 }
try {
    $c = Get-Content -LiteralPath $cache -Raw | ConvertFrom-Json
} catch {
    Write-Output "unreadable .comfy_launch.json: $_"; exit 1
}
if (-not $c -or -not $c.argv -or $c.argv.Count -eq 0) {
    Write-Output 'empty .comfy_launch.json'; exit 1
}
$cwd = $c.cwd
if (-not $cwd) {
    # Recover the cwd from the relative main.py argument: the first ancestor
    # of the exe where <ancestor>/<mainarg> actually exists.
    $mainarg = $c.argv | Where-Object { $_ -like '*main.py*' } | Select-Object -First 1
    if ($mainarg -and -not [IO.Path]::IsPathRooted($mainarg)) {
        $d = [IO.Path]::GetDirectoryName($c.argv[0])
        for ($i = 0; $i -lt 4 -and $d; $i++) {
            if (Test-Path -LiteralPath (Join-Path $d $mainarg)) { $cwd = $d; break }
            $d = [IO.Path]::GetDirectoryName($d)
        }
    }
}
if (-not $cwd) { Write-Output 'could not recover a working directory'; exit 1 }
$args = @()
if ($c.argv.Count -gt 1) { $args = $c.argv[1..($c.argv.Count - 1)] }
Write-Output "launching ComfyUI in $cwd"
Start-Process -FilePath $c.argv[0] -ArgumentList $args -WorkingDirectory $cwd -WindowStyle Hidden
exit 0
