import os
import gradio as gr
from gradio_iframe import iFrame
import torch
import gc
import functools

# Import modular components
import modules.print_pipeline as print_pipeline
from modules.content_moderation import get_content_moderator
from modules.gallery_manager import get_gallery_manager
from modules.three_d_viewer import create_3d_viewer_html
from modules.model_manager import get_model_manager
from modules.generation_pipeline import get_generation_pipeline

# Set environment variables
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'auto'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Global dictionary for storing per-user instances
user_instances = {}


# Device configuration helper
def get_device_config():
    """Get device configuration from environment or use defaults."""
    if os.environ.get("USE_MULTI_GPU") == "true":
        config = {
            "flux": os.environ.get("FLUX_DEVICE", "cuda:0"),
            "trellis": os.environ.get("TRELLIS_DEVICE", "cuda:1")
        }
        print(f"Multi-GPU mode enabled: FLUX on {config['flux']}, TRELLIS on {config['trellis']}")
        return config
    return None


def get_trellis_cpu_offload_setting():
    """Check if TRELLIS CPU offloading is requested."""
    return os.environ.get("ENABLE_TRELLIS_CPU_OFFLOAD") == "true"


def get_gaussian_rendering_setting():
    """Check if Gaussian rendering is requested."""
    return os.environ.get("USE_GAUSSIAN_RENDERING") == "true"


# Initialize cached managers
@functools.lru_cache(maxsize=1)
def load_model_manager():
    """Load model management system once and cache it."""
    device_config = get_device_config()
    enable_trellis_cpu_offload = get_trellis_cpu_offload_setting()
    image_model = os.environ.get('IMAGE_MODEL', 'flux')

    if enable_trellis_cpu_offload:
        print("🔄 TRELLIS CPU offloading enabled - this will reduce VRAM usage but may be slower")

    return get_model_manager(device_config, enable_trellis_cpu_offload, image_model=image_model)


@functools.lru_cache(maxsize=1)
def load_content_moderator():
    """Load content moderation system once and cache it."""
    return get_content_moderator()


@functools.lru_cache(maxsize=1)
def load_generation_pipeline():
    """Load generation pipeline system once and cache it."""
    return get_generation_pipeline()


@functools.lru_cache(maxsize=1)
def load_gallery_manager():
    """Load gallery management system once and cache it."""
    return get_gallery_manager()


def get_default_guidance_scale():
    """Get default guidance scale for the active image model."""
    model_manager = load_model_manager()
    config = model_manager.get_generation_config()
    return config["guidance_scale"]


def initialize_session(request: gr.Request):
    """Initialize user-specific session data."""
    # Initialize user session
    if request.session_hash not in user_instances:
        user_instances[request.session_hash] = {
            'flux_pipe': None,
            'trellis_pipeline': None,
            'reward_model': None,
            'generated_images': [],
            'selected_image_index': None,
            'video_path': None,
            'glb_path': None,
            'stl_file_path': None,
            'stl_file_number': None,
            'stl_prepared': False,
            'current_prompt': "",
            'generation_step': 0,  # 0: initial, 1: images generated, 2: 3D generated
            'use_gaussian_rendering': get_gaussian_rendering_setting(),
        }
        return "Session initialized"
    return "Session already exists"


def cleanup_session(request: gr.Request):
    """Clean up user session when they disconnect."""
    if request.session_hash in user_instances:
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        # Note: We keep user_instances for potential reconnection
        # but could delete if needed: del user_instances[request.session_hash]


def reset_generated_content(request: gr.Request):
    """Reset only generated content, preserving loaded models."""
    if request.session_hash in user_instances:
        session = user_instances[request.session_hash]
        session['generated_images'] = []
        session['selected_image_index'] = None
        session['video_path'] = None
        session['glb_path'] = None
        session['stl_file_path'] = None
        session['stl_file_number'] = None
        session['stl_prepared'] = False
        session['current_prompt'] = ""
        session['generation_step'] = 0

        torch.cuda.empty_cache()
        gc.collect()

        return (
            gr.update(value=[]),  # Clear gallery
            gr.update(value=""),  # Clear prompt
            "✅ Reset complete - ready for new project!"
        )

    return (gr.update(value=[]), gr.update(value=""), "Error: Session not found")


def load_models(request: gr.Request, progress=gr.Progress()):
    """Load AI models using the ModelManager."""
    if request.session_hash not in user_instances:
        return (
            "Error: Session not initialized",
            gr.update(visible=True),   # Keep loading section visible
            gr.update(visible=False)   # Keep generation section hidden
        )

    session = user_instances[request.session_hash]

    # Check if already loaded
    if session['flux_pipe'] is not None:
        return (
            "✅ Models already loaded!",
            gr.update(visible=False),  # Hide loading section
            gr.update(visible=True)    # Show generation section
        )

    progress(None, desc="🔄 Loading model manager...")

    # Load model manager
    model_manager = load_model_manager()

    # Load all models
    progress(None, desc="🔄 Loading AI models (this may take 1-2 minutes)...")
    flux_pipe, trellis_pipe, reward_model = model_manager.load_all_models()

    progress(None, desc="🔄 Finalizing...")

    # Store in session
    session['flux_pipe'] = flux_pipe
    session['trellis_pipeline'] = trellis_pipe
    session['reward_model'] = reward_model

    progress(None, desc="✅ Models loaded!")

    return (
        "✅ Models loaded successfully! You can now start creating.",
        gr.update(visible=False),  # Hide loading section
        gr.update(visible=True)    # Show generation section
    )


def generate_images(prompt: str, base_seed: int, prompt_suffix: str, guidance_scale: float, request: gr.Request, progress=gr.Progress()):
    """Generate images from prompt."""
    if request.session_hash not in user_instances:
        return None, "Error: Session not initialized"

    session = user_instances[request.session_hash]

    # Check if models loaded
    if session['flux_pipe'] is None:
        return None, "⚠️ Please load models first!"

    # Load managers
    content_moderator = load_content_moderator()
    generation_pipeline = load_generation_pipeline()
    model_manager = load_model_manager()

    # Content moderation check
    progress(None, desc="🛡️ Checking content safety...")
    is_safe, _scores = content_moderator.check_text_safety(prompt)

    if not is_safe:
        return None, "⚠️ Your prompt was flagged for potentially inappropriate content. Please modify your description."

    # Set pipeline models
    generation_pipeline.set_models(
        flux_pipeline=session['flux_pipe'],
        trellis_pipeline=session['trellis_pipeline'],
        reward_model=session['reward_model'],
        content_moderator=content_moderator
    )

    # Get generation config
    config = model_manager.get_generation_config()

    # Generate images
    progress(None, desc="🎨 Creating images from your description...")

    filtered_images = generation_pipeline.generate_images(
        prompt,
        num_images=4,  # Always generate 4 images (matches backend limit)
        base_seed=base_seed if base_seed > 0 else None,
        guidance_scale=guidance_scale,  # Use user-provided guidance scale
        num_inference_steps=config["num_inference_steps"],
        prompt_suffix=prompt_suffix  # Pass custom prompt suffix
    )

    if not filtered_images:
        return None, "⚠️ No images generated. Please try again."

    # Update session
    session['generated_images'] = filtered_images
    session['current_prompt'] = prompt
    session['generation_step'] = 1
    session['selected_image_index'] = None  # Clear previous selection

    progress(None, desc="✅ Images generated!")

    # Convert PIL images to format suitable for Gallery
    gallery_images = [(img, f"Image {i+1}") for i, img in enumerate(filtered_images)]

    return (
        gallery_images,  # Gallery value
        "✅ Images generated! Select an image below to generate 3D model."
    )


def select_and_generate_3d_for_index(image_index: int, model_seed: int, request: gr.Request, progress=gr.Progress()):
    """Handle image selection by index and generate 3D model."""
    if request.session_hash not in user_instances:
        use_gaussian = get_gaussian_rendering_setting()
        if use_gaussian:
            return None, None, "Error: Session not initialized"
        else:
            return None, "Error: Session not initialized"

    session = user_instances[request.session_hash]

    if not session['generated_images']:
        use_gaussian = get_gaussian_rendering_setting()
        if use_gaussian:
            return None, None, "Error: No images generated"
        else:
            return None, "Error: No images generated"

    if image_index >= len(session['generated_images']):
        use_gaussian = get_gaussian_rendering_setting()
        if use_gaussian:
            return None, None, f"Error: Image {image_index + 1} not available"
        else:
            return None, f"Error: Image {image_index + 1} not available"

    # Get selected image
    selected_idx = image_index
    session['selected_image_index'] = selected_idx
    selected_image = session['generated_images'][selected_idx]

    progress(None, desc="🔮 Preparing 3D generation...")

    # Load managers
    generation_pipeline = load_generation_pipeline()
    content_moderator = load_content_moderator()

    # Set pipeline models
    generation_pipeline.set_models(
        flux_pipeline=session['flux_pipe'],
        trellis_pipeline=session['trellis_pipeline'],
        reward_model=session['reward_model'],
        content_moderator=content_moderator
    )

    # Generate 3D model
    sample_video = session['use_gaussian_rendering']
    use_simple_glb = not session['use_gaussian_rendering']

    progress(None, desc="🔮 Transforming image into 3D model...")

    video_path, glb_path = generation_pipeline.generate_3d_model(
        selected_image,
        base_seed=model_seed if model_seed > 0 else None,
        sample_video=sample_video,
        use_simple_glb=use_simple_glb
    )

    # Update session
    session['video_path'] = video_path
    session['glb_path'] = glb_path
    session['generation_step'] = 2
    session['stl_prepared'] = False

    progress(None, desc="🎮 Generating 3D viewer...")

    # Generate viewer HTML with custom height
    viewer_html = create_3d_viewer_html(glb_path, container_height="550px")

    progress(None, desc="✅ 3D model ready!")

    # Return results based on rendering mode
    use_gaussian = session['use_gaussian_rendering']
    status_msg = f"✅ 3D model generated! Image #{selected_idx + 1} selected."

    if use_gaussian:
        video_output = video_path if video_path and len(video_path) > 0 else None
        return (
            video_output,  # video
            viewer_html,  # HTML viewer
            status_msg  # status
        )
    else:
        return (
            viewer_html,  # HTML viewer
            status_msg  # status
        )


def prepare_for_download(request: gr.Request):
    """Prepare 3D model for download and save to gallery."""
    if request.session_hash not in user_instances:
        return None, None, "Error: Session not initialized"

    session = user_instances[request.session_hash]

    if session['stl_prepared']:
        # Already prepared, just return the files
        return (
            session['stl_file_path'],
            session['glb_path'],
            f"✅ Saved as #{session['stl_file_number']}"
        )

    # Load gallery manager
    gallery_manager = load_gallery_manager()

    # Get next file number
    file_number = gallery_manager.increment_counter()
    output_dir = gallery_manager.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save GLB file
    glb_output = os.path.join(output_dir, f"model_{file_number}.glb")
    with open(glb_output, "wb") as f:
        with open(session['glb_path'], "rb") as f_in:
            f.write(f_in.read())

    # Generate STL
    stl_filepath = print_pipeline.run_with_file(glb_output, file_number, output_folder=output_dir)

    # Update session
    session['stl_file_path'] = stl_filepath
    session['stl_file_number'] = file_number
    session['stl_prepared'] = True

    # Save to gallery
    if session['selected_image_index'] is not None and session['generated_images']:
        selected_image = session['generated_images'][session['selected_image_index']]
        gallery_manager.save_item(
            file_number,
            session['current_prompt'],
            selected_image,
            stl_filepath,
            session['glb_path'],
            session['video_path']
        )

    return (
        stl_filepath,
        session['glb_path'],
        f"✅ Saved as #{file_number}"
    )


def get_gallery_items(search_query: str = ""):
    """Load gallery items."""
    gallery_manager = load_gallery_manager()
    return gallery_manager.load_items(search_query=search_query if search_query else None)


def get_next_print_number():
    """Get next print number."""
    gallery_manager = load_gallery_manager()
    return f"Next Print #: **{gallery_manager.get_next_print_number()}**"


def get_gallery_stats():
    """Get gallery statistics."""
    gallery_manager = load_gallery_manager()
    stats = gallery_manager.get_stats()
    return f"**{stats['total_items']}**"


# Compute default guidance scale once at startup
DEFAULT_GUIDANCE_SCALE = get_default_guidance_scale()

# Create Gradio interface
with gr.Blocks(
    title="Text-to-3D Generator",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
    .gradio-container {
        max-width: 1600px !important;
    }
    """
) as demo:

    # Title
    gr.Markdown("# 🎨 Text-to-3D Print Generator")
    gr.Markdown("Transform your ideas into printable 3D objects using AI")

    # Tabs
    with gr.Tabs():
        # Create Tab
        with gr.TabItem("🚀 Create"):
            # Two-column layout
            with gr.Row():
                # LEFT COLUMN - Inputs
                with gr.Column(scale=1):
                    # Model initialization (shown only if needed)
                    with gr.Column(visible=True) as loading_section:
                        load_models_btn = gr.Button(
                            "🚀 Initialize AI Models",
                            variant="primary",
                            size="lg"
                        )
                        loading_status = gr.Markdown()
                        gr.Markdown("""
                        **First time?** Click above to load AI models.
                        Takes 1-2 minutes, then instant generation!
                        """)

                    # Main input section (initially hidden)
                    with gr.Column(visible=False) as input_section:
                        # Prompt input
                        prompt_textbox = gr.Textbox(
                            label="✨ Describe your 3D object",
                            placeholder="E.g., 'A wise owl wearing glasses, holding a tiny book'\n\nPress Shift+Enter to generate",
                            lines=5
                        )

                        # Generate button
                        generate_btn = gr.Button(
                            "🎨 Generate Images",
                            variant="primary",
                            size="lg"
                        )

                        # Advanced settings
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            gr.Markdown("**Seeds** - Control randomness (0 = random)")
                            with gr.Row():
                                image_base_seed = gr.Number(
                                    label="Image Seed",
                                    value=0,
                                    minimum=0,
                                    maximum=999999,
                                    precision=0
                                )
                                model_base_seed = gr.Number(
                                    label="3D Seed",
                                    value=0,
                                    minimum=0,
                                    maximum=999999,
                                    precision=0
                                )

                            gr.Markdown("**Prompt Enhancement** - Customize 3D printing optimization")
                            prompt_suffix_input = gr.Textbox(
                                label="Prompt Suffix",
                                value=" Render of high quality 3D model on neutral background. Solid, contiguous mesh, optimized for 3D printing.",
                                lines=3,
                                placeholder="Text to append to your prompt for better 3D printing results"
                            )

                            gr.Markdown("**Guidance Scale (CFG)** - Control image generation strength")
                            guidance_scale_input = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=DEFAULT_GUIDANCE_SCALE,  # Set once at startup based on backend
                                info="Higher = more prompt adherence. FLUX: 0.0, Qwen: 2.0-5.5, Gemini: ignored"
                            )

                # RIGHT COLUMN - Outputs
                with gr.Column(scale=2):
                    # Generated images
                    gr.Markdown("### Generated Images")
                    gr.Markdown("**Click an image to generate 3D model**")
                    image_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=[2],
                        rows=[2],
                        height="500px",
                        allow_preview=False,
                        object_fit="contain",
                        visible=True
                    )

                    # 3D Model section
                    gr.Markdown("### 3D Model")

                    # Check if Gaussian rendering is enabled
                    use_gaussian = get_gaussian_rendering_setting()

                    if use_gaussian:
                        with gr.Row():
                            with gr.Column(scale=1):
                                video_output = gr.Video(label="Preview", visible=True)
                            with gr.Column(scale=2):
                                model_output = iFrame(height=550)
                    else:
                        model_output = iFrame(height=550)

                    # Download section
                    gr.Markdown("### Download")
                    with gr.Row():
                        download_stl_btn = gr.DownloadButton("📦 Download STL", visible=False)
                        download_glb_btn = gr.DownloadButton("📥 Download GLB", visible=False)
                    print_number_text = gr.Textbox(
                        label="Print Number",
                        interactive=False,
                        value=""
                    )

        # Gallery Tab
        with gr.TabItem("🖼️ Gallery"):
            gr.Markdown("## 🖼️ Gallery & History")

            with gr.Row():
                search_input = gr.Textbox(
                    label="🔍 Search by print number or prompt",
                    placeholder="e.g., 0001 or dragon",
                    scale=2
                )
                refresh_btn = gr.Button("🔄 Refresh Gallery")

            gallery_grid = gr.Gallery(
                label="Gallery",
                columns=4,
                height=600,
                allow_preview=True
            )

            gallery_status = gr.Markdown("Loading gallery...")

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by FLUX and TRELLIS
    </div>
    """, elem_classes=["footer"])

    # Event handlers

    # Session lifecycle
    demo.load(initialize_session, inputs=None, outputs=None)
    demo.unload(cleanup_session)

    # Model loading
    def load_and_show_input(request: gr.Request, progress=gr.Progress()):
        """Load models and show input section."""
        status, loading_vis, _ = load_models(request, progress)
        return (
            status,
            loading_vis,
            gr.update(visible=True)  # Show input section
        )

    load_models_btn.click(
        load_and_show_input,
        inputs=None,
        outputs=[loading_status, loading_section, input_section]
    )

    # Image generation
    def generate_images_for_gallery(prompt: str, seed: int, prompt_suffix: str, guidance_scale: float, request: gr.Request, progress=gr.Progress()):
        """Wrapper to generate images and return only gallery output."""
        gallery_images, status = generate_images(prompt, seed, prompt_suffix, guidance_scale, request, progress)
        # Use gr.update() to ensure gallery selection is reset
        return gr.update(value=gallery_images) if gallery_images else None

    generate_btn.click(
        generate_images_for_gallery,
        inputs=[prompt_textbox, image_base_seed, prompt_suffix_input, guidance_scale_input],
        outputs=[image_gallery]
    )

    # Also trigger on Enter key in prompt textbox
    prompt_textbox.submit(
        generate_images_for_gallery,
        inputs=[prompt_textbox, image_base_seed, prompt_suffix_input, guidance_scale_input],
        outputs=[image_gallery]
    )

    # Gallery selection handler - click on image to generate 3D
    def handle_gallery_select(evt: gr.SelectData, model_seed, request: gr.Request, progress=gr.Progress()):
        """Handle image selection from gallery and generate 3D."""
        image_index = evt.index  # Get selected image index from event

        # Call existing 3D generation function
        result = select_and_generate_3d_for_index(image_index, model_seed, request, progress)

        # Unpack based on Gaussian rendering
        use_gaussian = get_gaussian_rendering_setting()
        if use_gaussian:
            video, viewer_html, status = result
            return video, viewer_html, gr.update(visible=True), gr.update(visible=True), ""
        else:
            viewer_html, status = result
            return None, viewer_html, gr.update(visible=True), gr.update(visible=True), ""

    # Attach to gallery
    if get_gaussian_rendering_setting():
        image_gallery.select(
            handle_gallery_select,
            inputs=[model_base_seed],
            outputs=[video_output, model_output, download_stl_btn, download_glb_btn, print_number_text]
        ).then(
            prepare_for_download,
            inputs=None,
            outputs=[download_stl_btn, download_glb_btn, print_number_text]
        )
    else:
        image_gallery.select(
            handle_gallery_select,
            inputs=[model_base_seed],
            outputs=[gr.State(), model_output, download_stl_btn, download_glb_btn, print_number_text]
        ).then(
            prepare_for_download,
            inputs=None,
            outputs=[download_stl_btn, download_glb_btn, print_number_text]
        )

    # Gallery refresh
    def load_gallery(search=""):
        items = get_gallery_items(search)
        if not items:
            return [], "No items in gallery yet. Start creating to see them here!"

        # Convert to gallery format: list of image paths
        gallery_items = []
        for item in items:
            image_path = item[3]  # image_path is at index 3
            if os.path.exists(image_path):
                gallery_items.append((image_path, f"#{item[1]} - {item[2][:50]}"))

        return gallery_items, f"Found {len(items)} items"

    demo.load(
        load_gallery,
        inputs=None,
        outputs=[gallery_grid, gallery_status]
    )

    refresh_btn.click(
        load_gallery,
        inputs=search_input,
        outputs=[gallery_grid, gallery_status]
    )

    search_input.submit(
        load_gallery,
        inputs=search_input,
        outputs=[gallery_grid, gallery_status]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        max_file_size="50mb"
    )
