param([ValidateRange(1, 36500)][int]$RetainDays = 30, [switch]$DryRun)
. "$PSScriptRoot\scraper_common.ps1"
$operation = @('cleanup', '--retain-days', "$RetainDays")
if ($DryRun) { $operation += '--dry-run' }
Invoke-ScraperOperation -OperationArgs $operation
exit $script:OperationExitCode
