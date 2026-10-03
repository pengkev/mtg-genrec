param([Parameter(Mandatory = $true, Position = 0)][string]$BackupDirectory)
. "$PSScriptRoot\scraper_common.ps1"
Invoke-ScraperOperation -OperationArgs @('validate', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BackupDirectory))
exit $script:OperationExitCode
