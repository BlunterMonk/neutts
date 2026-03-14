python -m neutts_server --backbone-device gpu  --backbone "./neuphonic/neutts-air-Q8_0.gguf" --language en-us --host 192.168.1.198


# Get WSL IP
$wslIP = wsl hostname -I | ForEach-Object { $_.Trim().Split()[0] }
echo "WSL IP: $wslIP"

# Forward port (e.g., 50060)
netsh interface portproxy add v4tov4 listenport=50060 listenaddress=0.0.0.0 connectport=50060 connectaddress=$wslIP

# Allow through firewall
New-NetFirewallRule -DisplayName "WSL Port 50060" -Direction Inbound -Protocol TCP -LocalPort 50060 -Action Allow
