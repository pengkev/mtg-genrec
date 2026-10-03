param(
    [string]$StartTime = '18:00',
    [ValidateRange(0.01, 168)][double]$MaxRuntimeHours = 12,
    [switch]$WakeToRun,
    [string]$User = [Security.Principal.WindowsIdentity]::GetCurrent().Name,
    [string[]]$ScraperArguments = @()
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$taskName = 'MTG GenRec Scraper'
$marker = "MTG GenRec local scraper | $root"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.Description -ne $marker) { throw 'A task with this name belongs to another location; refusing overwrite.' }
$at = [DateTime]::ParseExact($StartTime, 'HH:mm', [Globalization.CultureInfo]::InvariantCulture)
$run = (Join-Path $PSScriptRoot 'run_scraper.ps1').Replace("'", "''")
$hours = $MaxRuntimeHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$command = "& '$run' -MaxRuntimeHours $hours"
foreach ($arg in $ScraperArguments) { $command += " '" + $arg.Replace("'", "''") + "'" }
$command += '; exit $LASTEXITCODE'
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $at
# The wrapper owns the time limit and bounded retries. Scheduler must not hard-kill
# a flushing writer or reset the wrapper's retry budget forever.
$settingsArgs = @{
    MultipleInstances = 'IgnoreNew'
    ExecutionTimeLimit = [TimeSpan]::Zero
    StartWhenAvailable = $true
    AllowStartIfOnBatteries = $true
    DontStopIfGoingOnBatteries = $true
}
if ($WakeToRun) { $settingsArgs.WakeToRun = $true }
$settings = New-ScheduledTaskSettingsSet @settingsArgs
$settings.AllowHardTerminate = $false
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $marker
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Write-Host "Installed '$taskName' for $User at $StartTime (max $hours hours). Not started. S4U may require local batch-logon rights; see operations guide."
