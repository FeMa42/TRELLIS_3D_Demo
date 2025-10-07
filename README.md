# Web Demo

This is an adaption of the original [TRELLIS repository](https://github.com/microsoft/TRELLIS). It adds a Streamlit web demo which combines FLUX-Dev/Schnell or Gemini-2.5-flash-image and TRELLIS. It also supports cpu offloading for the TRELLIS models to run them on GPUs with less VRAM.

Besides the standard TRELLIS installation using the setup.sh script, you also need to install streamlit and run the demo. You can install streamlit with the following command:
```sh
pip install streamlit
```

To use FLUX or gemini-2.5-flash-image you also need to install diffusers or google-genai package respectively. You can install them with the following commands:
```sh
pip install google-genai diffusers
```

Using FLUX-Dev is basically the setup we used at Gamescom 2025 as part of a demo for 3D generation. If you want to use Gemini-2.5-flash-image you have to also set your `GEMINI_API_KEY` environment variable: 

```sh
export GEMINI_API_KEY="your_api_key_here"
```

## running the demo

Best way to run the demo is using the provided `run_app.sh` script. It sets all necessary environment variables and starts the streamlit server. You can run it with the following command:
```sh
bash run_app.sh
```

Depending on your hardware setup you might want to adjust the environment variables in the `run_app.sh` script.

For setups with >= 24GB VRAM you can set `ENABLE_TRELLIS_CPU_OFFLOAD` and `ENABLE_FLUX_CPU_OFFLOAD` to false to get the best performance. For setups with less VRAM you should set them to true to enable cpu offloading. In my experiments it used less than 11 GB of VRAM with both offloadings enabled and I was able to use it with a NVIDIA RTX 4080 with 16GB of VRAM. If you have less VRAM you can also set `USE_GAUSSIAN_RENDERING` to false and enable Gemini-2.5-flash-image to further reduce the amount of needed memory. Without the rendering means, however, that you won't have a textured mesh and won't get the gaussian rendering option in the demo.

> CPU offloading for FLUX has a bigger impact on the amount of needed VRAM than for TRELLIS.
> The TRELLIS offloading is similar to the methods used in diffusers and each module is loaded on the GPU per demand. This was probably a bit overkill to be honest. It would have been enough to move all TRELLIS parts to cpu before and after each generation cycle.



## More information

More information about the installation and usage of TRELLIS can be found in the original [TRELLIS repository](https://github.com/microsoft/TRELLIS).
