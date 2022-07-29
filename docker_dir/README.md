# Build your image

You should build 2 images: one from `rl_py38_torch_dockerfile`, another from `core_dockerfile`.

## rl_py38_torch_dockerfile

This is an intermediate dockerfile. This image includes user declaration, conda installation, and basic python modules (torch, wandb, and more).

### Commands

You should put your name in ${PUT YOUR NAME} of below code snippet. Eg) guest-cch, guest-bjw

```
$ pwd
/home/lsj/apple-rl-core

$ docker build -f docker_dir/rl_py38_torch_dockerfile -t ${PUT YOUR NAME}/rl_py38_torch:0.1 \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=$(id -g) \
--build-arg USER_NAME=$(whoami) .
```

If this build is done, you can check your new docker image by `docker image ls` command.

## core_dockerfile

This is a child dockerfile for ml-core algorithm. This image only installs repository-specific conda environment on the top of the intermediate dockerfile.

### Commands

CAUTION: You should change *FROM lsj/rl_py38_torch:0.1* of **core_dockerfile** into the intermediate docker image you made using rl_py38_torch_dockerfile.

```
$ pwd
/home/lsj/apple-rl-core

$ cat docker_dir/core_dockerfile
FROM lsj/rl_py38_torch:0.1 # This should be changed, into like 'guest-cch/rl_py38_torch:0.1' or 'guest-bjw/rl_py38_torch:0.1'
...

$ docker build -f docker_dir/core_dockerfile -t ${PUT YOUR NAME}/core:0.1 . # Also, put your name again.
```
