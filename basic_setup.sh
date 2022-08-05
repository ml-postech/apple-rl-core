mkdir -p ${HOME}/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ${HOME}/.mujoco.tar.gz
tar -xf ${HOME}/.mujoco.tar.gz -C ${HOME}/.mujoco \
    && rm ${HOME}/.mujoco.tar.gz

echo 'export MUJOCO_GL=egl' >> ${HOME}/.bashrc
echo 'export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"' >> ${HOME}/.bashrc
echo 'export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"' >> ${HOME}/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia-000"' >> ${HOME}/.bashrc

source ${HOME}/.bashrc
conda env create --file conda_env.yml
