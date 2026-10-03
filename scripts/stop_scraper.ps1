param([ValidateRange(1, 86400)][int]$TimeoutSeconds = 120, [switch]$Force)
. "$PSScriptRoot\scraper_common.ps1"
$operation = @('stop', '--timeout', "$TimeoutSeconds")
if ($Force) { $operation += '--force' }
Invoke-ScraperOperation -OperationArgs $operation
exit $script:OperationExitCode
