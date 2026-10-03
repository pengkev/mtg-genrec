# Shared bootstrap. All entry points work from any current directory.
$ErrorActionPreference = 'Stop'
$script:ScraperRoot = Split-Path -Parent $PSScriptRoot
$script:ScraperPython = Join-Path $script:ScraperRoot '.venv\Scripts\python.exe'
$activation = Join-Path $script:ScraperRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path -LiteralPath $script:ScraperPython) -or -not (Test-Path -LiteralPath $activation)) {
    throw "Windows .venv is required. In '$script:ScraperRoot', run: py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-test.txt"
}
. $activation
function Invoke-ScraperOperation {
    param([string[]]$OperationArgs)
    Push-Location -LiteralPath $script:ScraperRoot
    try {
        & $script:ScraperPython -u -m scrape.operations @OperationArgs
        $script:OperationExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}
