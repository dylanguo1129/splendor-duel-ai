# Stop the background training loop (the GPU inference server exits with it). Progress is kept:
# start-training.ps1 resumes from train/runs/state.json.
$procs = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'loop\.mjs' }
if (-not $procs) { Write-Output "not running"; exit 0 }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force; Write-Output "stopped training loop pid $($p.ProcessId)" }
Start-Sleep 2
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'infer_server|train\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output "stopped $($_.ProcessId)" }
