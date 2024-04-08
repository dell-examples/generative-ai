# Created by Scalers AI for Dell Inc.

#!/bin/bash
sleep 5
python3 setup_influxdb.py
/bin/bash /entrypoint.sh telegraf
