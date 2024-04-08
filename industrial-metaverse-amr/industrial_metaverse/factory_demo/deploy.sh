# Created by Scalers AI for Dell Inc.

# install ffmpeg package
sudo apt-get install ffmpeg

# copy the factory demo code into NVIDIA® Isaac Sim™ user_examples directory
cp -r ./factory_demo/* /home/$USER/.local/share/ov/pkg/$ISAAC_SIM_PACKAGE/extension_examples/user_examples/

cd ..
export CONFIGDIR=$PWD/config

# install the necessary python packages
cd /home/$USER/.local/share/ov/pkg/$ISAAC_SIM_PACKAGE/
./python.sh -m pip install -r extension_examples/user_examples/requirements.txt

# start the NVIDIA® Isaac Sim™ on first GPU on the server.
CUDA_VISIBLE_DEVICES=0 ./isaac-sim.sh
