param([Parameter(Position = 0)][string]$BackupDirectory, [switch]$RecoverInterrupted)
. "$PSScriptRoot\scraper_common.ps1"
$operation = @('restore')
if ($RecoverInterrupted) {
    if ($BackupDirectory) { throw 'Use either BackupDirectory or RecoverInterrupted, not both.' }
    $operation += '--recover'
} elseif ($BackupDirectory) { $operation += $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BackupDirectory) }
else { throw 'Supply BackupDirectory, or use -RecoverInterrupted to roll back an interrupted restore.' }
Invoke-ScraperOperation -OperationArgs $operation
exit $script:OperationExitCode
