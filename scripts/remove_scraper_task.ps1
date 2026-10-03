$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$taskName = 'MTG GenRec Scraper'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) { Write-Host 'No scraper task installed.'; exit 0 }
if ($task.Description -ne "MTG GenRec local scraper | $root") { throw 'Task belongs to another repository; refusing removal.' }
if ($task.State -eq 'Running') { throw 'Stop the scraper with stop_scraper.ps1 before removing its task.' }
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
Write-Host 'Removed only this repository scraper task.'
