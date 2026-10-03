param([ValidateRange(0.01, 8760)][double]$StuckHours = 4)
. "$PSScriptRoot\scraper_common.ps1"
Invoke-ScraperOperation -OperationArgs @('health', '--stuck-hours', $StuckHours.ToString([Globalization.CultureInfo]::InvariantCulture))
exit $script:OperationExitCode
