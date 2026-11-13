$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker not found. Install Docker Desktop with Compose and start Docker.'
    }
    docker compose version
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose is unavailable. Install a current Docker Compose v2.' }
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Docker is unavailable. Start Docker and check access to its daemon.' }

    try {
        docker compose -p proj2697 -f compose.yaml up --build --wait --wait-timeout 120
        if ($LASTEXITCODE -ne 0) {
            docker compose -p proj2697 -f compose.yaml ps -a
            docker compose -p proj2697 -f compose.yaml logs --tail 80 backend
            throw 'Backend failed to start or /health did not become ready within 120 seconds. See details above.'
        }

        $url = 'http://localhost:8000/docs'
        Write-Host "Swagger: $url"
        Write-Host 'Press Ctrl+C to stop the backend and keep the data.'
        try { Start-Process $url }
        catch { Write-Warning "Could not open the browser. Open $url manually. $($_.Exception.Message)" }
        docker compose -p proj2697 -f compose.yaml logs --follow --tail 100 backend
        if ($LASTEXITCODE -ne 0) { throw 'Log streaming stopped unexpectedly. See details above.' }
    }
    finally {
        Write-Host 'Stopping backend (SQLite volume is preserved)...'
        docker compose -p proj2697 -f compose.yaml stop --timeout 10
        if ($LASTEXITCODE -ne 0) {
            Write-Warning 'Stop failed. Run: docker compose -p proj2697 -f compose.yaml stop'
        }
    }
}
finally { Pop-Location }
