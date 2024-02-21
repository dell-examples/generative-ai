# Created by Scalers AI for Dell Inc.

#!/bin/bash
sleep 5
python3 influx.py
/bin/bash /entrypoint.sh telegraf 
