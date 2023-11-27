# Floaty Removal 

This repository provides the implementation of our post-processing approach for floater removal in Neural Radiance Fields, specifically Instant-NGP.
> __[A Post Processing Technique to Automatically Remove Floater Artifacts in Neural Radiance Fields](https://diglib.eg.org/handle/10.1111/cgf14977)__  
> [Tristan Wirth](https://orcid.org/0000-0002-2445-9081), [Arne Rak](https://orcid.org/0000-0001-6385-3455), [Volker Knauthe](https://orcid.org/0000-0001-6993-5099), [Dieter W. Fellner](https://orcid.org/0000-0001-7756-0901)  
> _Computer Graphics Forum v42, 2023_  

A Dockerfile is provided for ease of use.

## Prerequisites

1. NVIDIA GPU (RTX 20 Series and above)
2. CUDA Toolkit (11.2+)
3. [Docker and Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
4. For evaluation, you may download the [Nerfbusters dataset](https://github.com/ethanweber/nerfbusters) and insert it into the dataset folder. It will be mounted in the docker container.

## Floaty Removal Docker Container
```
# build image
docker compose -f ./docker-compose.yml build floatyremoval
# optional if Instant-NGP GUI is to be used
xhost local:root
# open bash inside the container
docker compose -f ./docker-compose.yml run floatyremoval /bin/bash
```

### Run colmap

From the Docker container's bash, run the `colmap2nerf.py` script to generate a `transforms.json` file for Instant-NGP.

```
cd dataset/pikachu
python3 /opt/instant-ngp/scripts/colmap2nerf.py --run_colmap --images images_2
```

### Post-processing density grid

Example usage of our post-processing approach for floater removal is shown in `eval.py`.

```
cd /volume
python3 eval.py dataset/pikachu
```

# Citation

```
@article {10.1111:cgf.14977,
    journal = {Computer Graphics Forum},
    title = {{A Post Processing Technique to Automatically Remove Floater Artifacts in Neural Radiance Fields}},
    author = {Wirth, Tristan and Rak, Arne and Knauthe, Volker and Fellner, Dieter W.},
    year = {2023},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.14977}
}
```