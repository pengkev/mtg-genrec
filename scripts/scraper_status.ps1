. "$PSScriptRoot\scraper_common.ps1"
Invoke-ScraperOperation -OperationArgs @('status')
exit $script:OperationExitCode
