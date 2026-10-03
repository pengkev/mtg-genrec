param([string]$Destination)
. "$PSScriptRoot\scraper_common.ps1"
$operation = @('backup')
if ($Destination) { $operation += @('--destination', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)) }
Invoke-ScraperOperation -OperationArgs $operation
exit $script:OperationExitCode
