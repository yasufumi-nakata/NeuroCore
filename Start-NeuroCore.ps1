param(
  [string]$SshHost = "smac20",
  [string]$SshUser = "your_ssh_username",
  [int]$ModelPort = 8012,
  [int]$TunnelPort = 18010,
  [int]$AppPort = 8788,
  [switch]$NoTunnel
)
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run Install-NeuroCore.ps1 first" }
$sshProcess = $null
try {
  if (-not $NoTunnel) {
    $forward = "127.0.0.1:$($TunnelPort):127.0.0.1:$($ModelPort)"
    $sshArgs = @("-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "StrictHostKeyChecking=yes", "-N", "-L", $forward, "$SshUser@$SshHost")
    $sshProcess = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -PassThru -NoNewWindow
    Start-Sleep -Seconds 2
    if ($sshProcess.HasExited) { throw "SSH failed. Check VPN, SSH key, and known_hosts" }
  }
  Write-Host "Open http://127.0.0.1:$AppPort"
  & $python -m neurocore.cli serve --desktop --remote-model "http://127.0.0.1:$TunnelPort" --port $AppPort
  if ($LASTEXITCODE -ne 0) { throw "NeuroCore exited: $LASTEXITCODE" }
} finally {
  if ($null -ne $sshProcess -and -not $sshProcess.HasExited) {
    Stop-Process -Id $sshProcess.Id -ErrorAction SilentlyContinue
  }
}
