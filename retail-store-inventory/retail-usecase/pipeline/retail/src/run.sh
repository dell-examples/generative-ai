export TF_ENABLE_ZENDNN_OPTS=0
export ZENDNN_CONV_ALGO=3
export ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
export ZENDNN_TENSOR_POOL_LIMIT=512
export OMP_NUM_THREADS=32
python3 retail_streams.py --config /src/config/config.yaml