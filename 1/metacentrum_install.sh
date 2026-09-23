# Here, variables for the Conda environment and Jupyter kernel names are set.
# Use only lowercase characters!!

KERNEL_NAME="pli_en_2025_v2"
ENV_NAME="pli_en_2025_v2"

# Navigate to the user's home directory
cd $HOME

# Create directories for the Conda environment and temporary files if they don't already exist
mkdir -p envs
mkdir -p temp

# Set the environment variable for temporary files
export TMPDIR=/storage/brno2/home/$USER/temp

# Add Mambaforge to available modules
module add mambaforge

# Create a new Conda environment with Python, IPython Kernel, Jupyter, Torch, etc.
mamba create -y --prefix /storage/brno2/home/$USER/envs/$ENV_NAME python ipykernel transformers tqdm scikit-learn matplotlib jupyter pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Add the IPython Kernel module
module add py-ipykernel

# Install the Jupyter kernel for the user. Sets the kernel name and display name.
python3 -m ipykernel install --user --name $KERNEL_NAME --display-name $KERNEL_NAME

# Set the path to the file that will contain the script for launching the kernel
filename="/storage/brno2/home/${USER}/.local/share/jupyter/kernels/${KERNEL_NAME}/start_kernel.sh"

# Prepare the script for starting the kernel
script_text='#!/bin/bash'

# Script to activate the Conda environment and launch the IPython kernel with the passed arguments
script_text+="
exec /storage/brno2/home/${USER}/envs/${ENV_NAME}/bin/python \"\$@\"
"

# Write the script to the file
echo "$script_text" > "$filename"

# Set executable permissions for the script
chmod a+x /storage/brno2/home/$USER/.local/share/jupyter/kernels/$KERNEL_NAME/start_kernel.sh

# Modify the kernel.json file to use the prepared script when launching the kernel
sed -i "3s|\"[^\"]*\"|\"$filename\"|" /storage/brno2/home/$USER/.local/share/jupyter/kernels/$KERNEL_NAME/kernel.json

echo -e "###### OnDemand setup completed ######\nTo activate the environment, use:\nmamba activate \e[32m/storage/brno2/home/${USER}/envs/$ENV_NAME\e[0m"