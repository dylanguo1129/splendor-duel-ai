# Start the training loop as an independent background process (survives the terminal / Claude
# session closing). It gets its own process group and no window, so console Ctrl+C / Ctrl+Break
# events from other programs cannot reach it. Stop it with scripts/stop-training.ps1.
param(
  [string]$LoopArgs = "--gpu 1 --hidden 512,512 --workers 24 --concurrency 40 --gpu-servers 3 --min-free-gb 4 --hard-free-gb 1.5 --max-rss-gb 10 --gens 200 --games 3000 --sims 600 --sims-fast 120 --fpur 0.3 --root-fpur 0 --aux-weight 0.1 --entity 1 --gate-games 400 --gate-sims 200 --gate-threshold 0.53 --gate-extra 800 --gate-extra-threshold 0.515 --epochs 2 --finetune-lr 1.5e-4 --holdout w23 --window 8 --league 1 --p-self 0.5 --old-count 3 --style-bench-games 100",
  [switch]$SelfPlayFirst # resume with self-play instead of retraining the last generation (after a format change)
)
if ($SelfPlayFirst) { $LoopArgs += " --selfplay-first 1" }
$dir = Split-Path -Parent $PSScriptRoot
$running = Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object { $_.CommandLine -match 'loop\.mjs' }
if ($running) { Write-Output "already running: pid $($running.ProcessId)"; exit 0 }
$node = (Get-Command node).Source
$CREATE_NEW_PROCESS_GROUP = 0x00000200
$CREATE_NO_WINDOW = 0x08000000
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0; CreateFlags = [uint32]($CREATE_NEW_PROCESS_GROUP -bor $CREATE_NO_WINDOW) }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "`"$node`" dist/train/loop.mjs $LoopArgs"
  CurrentDirectory = $dir
  ProcessStartupInformation = $startup
}
if ($r.ReturnValue -ne 0) { throw "failed to start (WMI code $($r.ReturnValue))" }
Write-Output "training loop started: pid $($r.ProcessId) (log: train/runs/log.txt)"
