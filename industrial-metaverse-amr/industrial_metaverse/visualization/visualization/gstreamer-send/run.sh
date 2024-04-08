# Created by Scalers AI for Dell Inc.

export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gstreamer-1.0
while true
do
go run main.go
echo "Restarting WebRTC stream.. Looking for new offer from browser"
sleep 1
done
