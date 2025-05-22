$projectPath = "C:/Users/User/Desktop/PythonAICourse/Modul3/Curs1/proiect/crypto_model"
Set-Location $projectPath

Start-Process -NoNewWindow -FilePath "mlflow" -ArgumentList "server", "--host", "127.0.0.1", "--port", "5003"
Start-Sleep -Seconds 10

$ngrokLogPath = "$env:TEMP\ngrok_log.txt"
Start-Process -FilePath "ngrok" -ArgumentList "http", "5003", "--log=stdout" `
    -RedirectStandardOutput $ngrokLogPath `
    -NoNewWindow

$ngrokUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
        foreach ($tunnel in $response.tunnels) {
            if ($tunnel.public_url -like "https://*") {
                $ngrokUrl = $tunnel.public_url
                break
            }
        }
    } catch {
        Start-Sleep -Seconds 1
        continue
    }
    if ($ngrokUrl) { break }
    Start-Sleep -Seconds 1
}

if (-not $ngrokUrl) {
    Write-Error "Failed to retrieve ngrok URL after waiting"
    exit 1
}

Write-Host "Ngrok URL: $ngrokUrl"

$dockerTemplatePath = "$projectPath/docker-compose.template.yml"
$dockerGeneratedPath = "$projectPath/docker-compose.generated.yml"

(Get-Content $dockerTemplatePath) -replace "NGROK_URL_PLACEHOLDER", $ngrokUrl | Set-Content $dockerGeneratedPath

docker-compose -f $dockerGeneratedPath up -d
