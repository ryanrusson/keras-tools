# keras-tools
Description: Repo for tools developed to help with deep learning with Keras. Also contains info on how to make sure
 you are up and running with Docker + Nvidia GPUs

## Tips for ensuring Docker is setup to use Nvidia GPU
1. Install Docker using the instructions found here (Ubuntu): https://docs.docker.com/engine/install/ubuntu/#installation-methods
2. Install Nvidia container runtime packages
    1. Visit https://nvidia.github.io/nvidia-container-runtime/ and follow the instructions for your installation to
     add the Nvidia repos
    2. Visit https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu and follow these
     instructions to get the runtime environment installed
    3. Restart Docker `systemctl restart docker` on Ubuntu systems
3. Ensure the above worked by running `sudo docker run -it --rm --gpus all ubuntu nvidia-smi` which should output
 something like this:  
 ![nvidia-smi screenshot](assets/images/nvidiasmi-keras-tools.png)
4. Retrieve an image of interest from the Nvidia GPU Cloud: https://ngc.nvidia.com/catalog/
    1. NOTE: This requires a login to NGC
    2. Follow the instructions to get the image within the docs for the image of interest
5. Ensure your drivers are compatible with the image you selected. If not, you will get a warning like this when you
 spin up the container: `ERROR: This container was built for NVIDIA Driver Release 450.51 or later, but
       version 430.50 was detected and compatibility mode is UNAVAILABLE.`
    1. (Ubuntu) Search for the desired driver by using `apt search nvidia-driver`
    2. (Ubuntu) Install the driver `sudo apt install nvidia-driver-450` for example
    3. Reboot your machine `sudo reboot`
    4. Verify the install after reboot with `nvidia-smi`
 

    