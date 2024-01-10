curl --request POST \
  http://localhost:8086/api/v2/query?org=scalers \
  --header 'Authorization: Token gTnHDvKKEsgUn6Xp5cC9Z3at6WZG_GCSAQ0Ob-HroTDnnHuL-sITT_p2Q3TpS99AAmS_JreMnHXqQ3dV1jIUaw==' \
  --header 'Accept: application/csv' \
  --header 'Content-type: application/vnd.flux' \
  --data 'from(bucket: "telegraf")
  |> range(start: -1000h)
  |> filter(fn: (r) => r["_measurement"] == "cpu")
  |> filter(fn: (r) => r["_field"] == "usage_system")
  |> filter(fn: (r) => r["cpu"] == "cpu-total")
  |> filter(fn: (r) => r["host"] == "76655002595c")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "mean")' >> cpu_usage.csv

curl --request POST \
  http://localhost:8086/api/v2/query?org=scalers \
  --header 'Authorization: Token gTnHDvKKEsgUn6Xp5cC9Z3at6WZG_GCSAQ0Ob-HroTDnnHuL-sITT_p2Q3TpS99AAmS_JreMnHXqQ3dV1jIUaw==' \
  --header 'Accept: application/csv' \
  --header 'Content-type: application/vnd.flux' \
  --data 'from(bucket: "telegraf")
  |> range(start: -1000h)
  |> filter(fn: (r) => r["_measurement"] == "mem")
  |> filter(fn: (r) => r["_field"] == "used")
  |> filter(fn: (r) => r["host"] == "76655002595c")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "mean")' >> memory_usage.csv

curl --request POST \
  http://localhost:8086/api/v2/query?org=scalers \
  --header 'Authorization: Token gTnHDvKKEsgUn6Xp5cC9Z3at6WZG_GCSAQ0Ob-HroTDnnHuL-sITT_p2Q3TpS99AAmS_JreMnHXqQ3dV1jIUaw==' \
  --header 'Accept: application/csv' \
  --header 'Content-type: application/vnd.flux' \
  --data 'from(bucket: "telegraf")
  |> range(start: -1000h)
  |> filter(fn: (r) => r["_measurement"] == "gpu_stats")
  |> filter(fn: (r) => r["_field"] == "mem_util")
  |> filter(fn: (r) => r["device"] == "0" or r["device"] == "1" or r["device"] == "2" or r["device"] == "3" or r["device"] == "4" or r["device"] == "5" or r["device"] == "6" or r["device"] == "7")
  |> filter(fn: (r) => r["host"] == "76655002595c")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "mean")' >> gpu_memory_usage.csv

curl --request POST \
  http://localhost:8086/api/v2/query?org=scalers \
  --header 'Authorization: Token gTnHDvKKEsgUn6Xp5cC9Z3at6WZG_GCSAQ0Ob-HroTDnnHuL-sITT_p2Q3TpS99AAmS_JreMnHXqQ3dV1jIUaw==' \
  --header 'Accept: application/csv' \
  --header 'Content-type: application/vnd.flux' \
  --data 'from(bucket: "telegraf")
  |> range(start: -1000h)
  |> filter(fn: (r) => r["_measurement"] == "gpu_stats")
  |> filter(fn: (r) => r["_field"] == "comp_util")
  |> filter(fn: (r) => r["device"] == "0" or r["device"] == "1" or r["device"] == "2" or r["device"] == "3" or r["device"] == "4" or r["device"] == "5" or r["device"] == "6" or r["device"] == "7")
  |> filter(fn: (r) => r["host"] == "76655002595c")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "mean")' >> gpu_computational_usage.csv
