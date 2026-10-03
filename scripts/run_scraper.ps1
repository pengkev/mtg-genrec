# Deliberately parse only wrapper options: unknown arguments pass through verbatim,
# including --sources followed by multiple values on Windows PowerShell 5.1.
. "$PSScriptRoot\scraper_common.ps1"
$options = @{
    '-MaxRuntimeHours' = '--max-runtime-hours'
    '-MinimumFreeGB' = '--minimum-free-gb'
    '-CriticalFreeGB' = '--critical-free-gb'
    '-DiskCheckSeconds' = '--disk-check-seconds'
    '-ShutdownTimeoutSeconds' = '--shutdown-timeout'
    '-MaxRetries' = '--max-retries'
}
$operation = @('run', '--wrapper-pid', "$PID")
$forward = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $value = [string]$args[$i]
    if ($value -eq '--') {
        for ($j = $i + 1; $j -lt $args.Count; $j++) { $forward += [string]$args[$j] }
        break
    }
    if ($options.ContainsKey($value)) {
        if ($i + 1 -ge $args.Count) { throw "Missing value for $value" }
        $operation += $options[$value]
        $i++
        $operation += [string]$args[$i]
    } else { $forward += $value }
}
Invoke-ScraperOperation -OperationArgs ($operation + @('--') + $forward)
exit $script:OperationExitCode
