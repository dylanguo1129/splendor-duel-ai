# Restrict the running training loop (and its GPU inference server) to the efficiency cores, so the
# performance cores stay free for other work (e.g. playing against the AI in the app); or give it all cores
# back. Takes effect immediately, no restart. Processes the loop starts later (train.py, a restarted
# inference server) inherit the setting. A fresh start-training.ps1 starts on all cores again.
#   powershell -ExecutionPolicy Bypass -File scripts/set-training-cores.ps1 -Cores ecores
#   powershell -ExecutionPolicy Bypass -File scripts/set-training-cores.ps1 -Cores all
param([ValidateSet("ecores", "all")][string]$Cores = "ecores")

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CpuSets {
  [DllImport("kernel32.dll", SetLastError = true)]
  static extern bool GetSystemCpuSetInformation(IntPtr info, int len, out int retLen, IntPtr proc, int flags);
  // Bit mask of the logical processors with the lowest efficiency class (E-cores on hybrid CPUs).
  public static long EfficiencyMask() {
    int len;
    GetSystemCpuSetInformation(IntPtr.Zero, 0, out len, IntPtr.Zero, 0);
    IntPtr buf = Marshal.AllocHGlobal(len);
    try {
      if (!GetSystemCpuSetInformation(buf, len, out len, IntPtr.Zero, 0)) return 0;
      long[] masks = new long[256];
      int minEff = 255, maxEff = 0, off = 0;
      while (off < len) {
        int size = Marshal.ReadInt32(buf, off);
        int lp = Marshal.ReadByte(buf, off + 14), eff = Marshal.ReadByte(buf, off + 18);
        if (lp < 64) masks[eff] |= 1L << lp;
        minEff = Math.Min(minEff, eff); maxEff = Math.Max(maxEff, eff);
        off += size;
      }
      return minEff == maxEff ? 0 : masks[minEff];
    } finally { Marshal.FreeHGlobal(buf); }
  }
}
'@

$all = [int64]([math]::Pow(2, [Environment]::ProcessorCount) - 1)
$mask = $all
if ($Cores -eq "ecores") {
  $mask = [CpuSets]::EfficiencyMask()
  if ($mask -eq 0) { Write-Output "this CPU has no separate efficiency cores; nothing to do"; exit 0 }
}
$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -eq "node.exe" -and $_.CommandLine -match "loop\.mjs") -or
  ($_.Name -eq "python.exe" -and $_.CommandLine -match "infer_server|train\.py")
}
if (-not $procs) { Write-Output "training is not running"; exit 0 }
foreach ($p in $procs) {
  (Get-Process -Id $p.ProcessId).ProcessorAffinity = [IntPtr]$mask
  Write-Output ("{0} pid {1}: affinity 0x{2:X}" -f $p.Name, $p.ProcessId, $mask)
}
