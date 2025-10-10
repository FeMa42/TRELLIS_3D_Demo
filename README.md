# Web Demo

This is an adaption of the original [TRELLIS repository](https://github.com/microsoft/TRELLIS). It adds a Gradio web demo which combines FLUX-Dev/Schnell, Qwen-Image or Gemini-2.5-flash-image with TRELLIS to enable text to 3D generation. It also supports cpu offloading for FLUX, QWEN and TRELLIS to run them on GPUs with less VRAM.

You only need to install the dependencies in the requirements.txt file. You can install them with the following command:
```sh
pip install -r requirements.txt
```
I tested it inside a conda environment with python 3.10 on Ubuntu 22.04 LTS with Nvidia Driver Version: 570.172.08 for CUDA 12.8. 

If you want to use Gemini-2.5-flash-image you have to also set your `GEMINI_API_KEY` environment variable: 
```sh
export GEMINI_API_KEY="your_api_key_here"
```

## Running the Text to 3D Demo

Best way to run the demo is using the provided `app_text.sh` script. It sets all necessary environment variables and starts the streamlit server. You can run it with the following command:
```sh
./app_text.sh
```

Depending on your hardware setup you might want to adjust the environment variables in the `app_text.sh` script.

For setups with >= 40GB VRAM you can set `ENABLE_TRELLIS_CPU_OFFLOAD` and `ENABLE_IMAGE_CPU_OFFLOAD` to false to get the best performance. For setups with less VRAM you should set them to true to enable cpu offloading. In my experiments TRELLIS used less than 11 GB of VRAM with both offloadings enabled and I was able to use it with a NVIDIA RTX 4080 with 16GB of VRAM. If you have less VRAM you can also set `USE_GAUSSIAN_RENDERING` to false and enable Gemini-2.5-flash-image to further reduce the amount of needed memory. Without the rendering, you won't get a textured mesh and won't get the gaussian rendering option in the demo.

> CPU offloading for FLUX and Qwen has a bigger impact on the amount of needed VRAM than for TRELLIS.
> The TRELLIS CPU offloading is similar to the methods used in diffusers and each module is loaded on the GPU per demand. This was probably a bit overkill to be honest. It would have been enough to move all TRELLIS parts to GPU before and off after each generation cycle at once.

## Running Image to 3D Demo

The text to 3D demo is basically the same as in the hunnginface space [trellis-community/TRELLIS](https://huggingface.co/spaces/trellis-community/TRELLIS). You can run it using the 'app.py' script. You can start it with the following command:
```sh
python app.py
```

## More information

Using the Streamlit demo with FLUX-Dev provided `run_app.sh` is basically the setup we used at Gamescom 2025 as part of a demo for 3D generation. For this you also need to install streamlit. 

More information about the installation, training, data and usage of TRELLIS can be found in the original [TRELLIS repository](https://github.com/microsoft/TRELLIS).
