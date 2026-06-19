import os
import gradio as gr
from gradio_iframe import iFrame
import torch
import gc
import functools

# Import modular components
from modules.content_moderation import get_content_moderator
from modules.gallery_manager import get_gallery_manager
from modules.three_d_viewer import create_3d_viewer_html
from modules.model_manager import get_model_manager
from modules.generation_pipeline import get_generation_pipeline

# Set environment variables
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'  # verified-working spconv algo on Blackwell (sm_120)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Printability (Stage-1) -- default ON, env-overridable. See printability_optimization_3d.md.
# Fills enclosed voids in the Stage-1 occupancy grid before Stage-2: ~15% less print
# support material, detail-safe. setdefault so run_streamlit_app.sh / operators can override.
os.environ.setdefault('TRELLIS_STAGE1_FILL_HOLES', 'true')
# Optional Stage-1 DPO LoRA (off by default). To enable, point at the vendored checkpoint:
#   export TRELLIS_STAGE1_LORA=checkpoints/printability_lora_r16
os.environ.setdefault('TRELLIS_STAGE1_LORA', '')
# Printable STL export: voxel-remesh into a single watertight solid on STL export
# (validated in investigations/mesh_repair). env-overridable; set 'off' to disable.
os.environ.setdefault('TRELLIS_PRINT_REMESH', 'voxel288_smooth')
# Decimate the print-ready mesh to this target face count (lighter viewer + slicer; 0 disables).
os.environ.setdefault('TRELLIS_PRINT_TARGET_FACES', '25000')

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
            'gallery_items_to_show': 20,  # Number of gallery items to display
            'gallery_view_mode': 'Grid',  # Grid or Compact view
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
        return None, None, "Error: Session not initialized"

    session = user_instances[request.session_hash]

    if not session['generated_images']:
        return None, None, "Error: No images generated"

    if image_index >= len(session['generated_images']):
        return None, None, f"Error: Image {image_index + 1} not available"

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

    progress(None, desc="🔮 Transforming image into 3D model...")

    glb_path, print_ready_glb_path, stl_path = generation_pipeline.generate_3d_model(
        selected_image,
        base_seed=model_seed if model_seed > 0 else None
    )

    # Update session
    session['glb_path'] = glb_path
    session['stl_file_path'] = stl_path
    session['generation_step'] = 2
    session['stl_prepared'] = False

    progress(None, desc="🎮 Generating 3D viewers...")

    # Build colored viewer HTML
    colored_html = create_3d_viewer_html(glb_path, container_height="550px")

    # Build normals viewer HTML (or placeholder when unavailable)
    if print_ready_glb_path:
        normals_html = create_3d_viewer_html(print_ready_glb_path, normals=True, container_height="550px")
    else:
        normals_html = (
            "<div style='height:550px;display:flex;align-items:center;"
            "justify-content:center;color:#888'>No printable mesh available</div>"
        )

    progress(None, desc="✅ 3D model ready!")

    status_msg = f"✅ 3D model generated! Image #{selected_idx + 1} selected."
    return colored_html, normals_html, status_msg


def prepare_for_download(request: gr.Request):
    """Prepare 3D model for download and save to gallery."""
    if request.session_hash not in user_instances:
        return None, None, "Error: Session not initialized"

    session = user_instances[request.session_hash]

    if session['stl_prepared']:
        # Already prepared, just return the files
        stl_path = session['stl_file_path']
        return (
            stl_path if stl_path and os.path.exists(stl_path) else gr.update(visible=False),
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

    # Use STL from generation (already produced by generate_3d_model)
    stl_filepath = session.get('stl_file_path')

    # Update session
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
            None  # no video
        )

    stl_download = stl_filepath if stl_filepath and os.path.exists(stl_filepath) else gr.update(visible=False)
    return (
        stl_download,
        glb_output,
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
                    gr.Markdown("**Colored** (left) · **Print-ready normals** (right)")

                    with gr.Row():
                        model_output_colored = iFrame(height=550)
                        model_output_normals = iFrame(height=550)

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

            # Controls row
            with gr.Row():
                search_input = gr.Textbox(
                    label="🔍 Search by print number or prompt",
                    placeholder="e.g., 0001 or dragon",
                    scale=2
                )
                view_mode = gr.Radio(
                    label="View Mode",
                    choices=["Grid", "Compact"],
                    value="Grid",
                    scale=1
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=0.5)

            gallery_status = gr.Markdown("Loading gallery...")

            # Hidden state for tracking pagination
            items_to_show_state = gr.State(20)

            # Grid View (improved scrollable gallery with details)
            with gr.Column(visible=True) as grid_view:
                # Main gallery grid - now scrollable
                gallery_grid_view = gr.Gallery(
                    label="Gallery Items",
                    columns=4,
                    height="800px",  # Increased height for better scrolling
                    allow_preview=True,
                    object_fit="contain"
                )

                # Pagination control
                with gr.Row():
                    items_shown_text = gr.Markdown("Showing 0 items")
                    load_more_btn = gr.Button("📥 Load More Items", visible=False, scale=0.5)

                # Item details section (shown when item is selected)
                with gr.Accordion("📋 Item Details", open=False, visible=False) as item_details_accordion:
                    with gr.Row():
                        with gr.Column(scale=1):
                            detail_image = gr.Image(label="Preview", interactive=False)
                        with gr.Column(scale=1):
                            detail_info = gr.Markdown("No item selected")
                            with gr.Row():
                                detail_download_stl = gr.DownloadButton("📦 Download STL", visible=False)
                                detail_download_glb = gr.DownloadButton("📥 Download GLB", visible=False)

                    # 3D Viewer in details
                    detail_viewer = iFrame(height=500, visible=False)

            # Compact View (simple gallery without details)
            with gr.Column(visible=False) as compact_view:
                gallery_grid_compact = gr.Gallery(
                    label="Gallery",
                    columns=4,
                    height="800px",
                    allow_preview=True,
                    object_fit="contain"
                )

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

        colored_html, normals_html, status = select_and_generate_3d_for_index(
            image_index, model_seed, request, progress
        )
        return colored_html, normals_html, gr.update(visible=True), gr.update(visible=True), ""

    # Attach to gallery
    image_gallery.select(
        handle_gallery_select,
        inputs=[model_base_seed],
        outputs=[model_output_colored, model_output_normals, download_stl_btn, download_glb_btn, print_number_text]
    ).then(
        prepare_for_download,
        inputs=None,
        outputs=[download_stl_btn, download_glb_btn, print_number_text]
    )

    # Gallery loading functions
    def load_gallery_with_pagination(search="", items_to_show=20, view_mode="Grid"):
        """Load gallery items with pagination support."""
        items = get_gallery_items(search)

        if not items:
            status = "No items in gallery yet. Start creating to see them here!"
            return (
                [],  # gallery_grid_view
                [],  # gallery_grid_compact
                status,  # gallery_status
                "Showing 0 items",  # items_shown_text
                gr.update(visible=False),  # load_more_btn
                gr.update(visible=(view_mode == "Grid")),  # grid_view
                gr.update(visible=(view_mode == "Compact"))  # compact_view
            )

        # Limit items based on pagination
        total_items = len(items)
        items_to_display = items[:items_to_show]

        # Convert to gallery format: list of (image_path, caption) tuples
        gallery_items = []
        for item in items_to_display:
            image_path = item[3]  # image_path is at index 3
            if os.path.exists(image_path):
                caption = f"#{item[1]} - {item[2][:50]}"
                gallery_items.append((image_path, caption))

        # Status and pagination info
        status = f"Found {total_items} items"
        items_shown = f"Showing {len(gallery_items)} of {total_items} items"
        show_load_more = len(gallery_items) < total_items

        return (
            gallery_items,  # gallery_grid_view
            gallery_items,  # gallery_grid_compact
            status,  # gallery_status
            items_shown,  # items_shown_text
            gr.update(visible=show_load_more),  # load_more_btn
            gr.update(visible=(view_mode == "Grid")),  # grid_view
            gr.update(visible=(view_mode == "Compact"))  # compact_view
        )

    def handle_view_mode_change(view_mode):
        """Toggle between Grid and Compact view modes."""
        return (
            gr.update(visible=(view_mode == "Grid")),  # grid_view
            gr.update(visible=(view_mode == "Compact"))  # compact_view
        )

    def handle_load_more(search="", view_mode="Grid", current_items_to_show=20):
        """Load more items when button is clicked."""
        new_items_to_show = current_items_to_show + 20
        result = load_gallery_with_pagination(search, new_items_to_show, view_mode)
        # Add the new items count as last return value
        return result + (new_items_to_show,)

    def handle_gallery_item_select(evt: gr.SelectData, search=""):
        """Handle item selection from gallery and show details."""
        selected_index = evt.index

        # Get all items to find the selected one
        items = get_gallery_items(search)

        if selected_index >= len(items):
            return (
                None,  # detail_image
                "Error: Item not found",  # detail_info
                None,  # detail_download_stl
                None,  # detail_download_glb
                gr.update(visible=False),  # detail_viewer
                gr.update(visible=False, open=False)  # item_details_accordion
            )

        item = items[selected_index]
        # item format: (id, print_number, prompt, image_path, stl_path, glb_path, video_path, created_at, metadata)

        # Prepare detail info
        detail_text = f"""
**Print Number:** #{item[1]}

**Prompt:** {item[2]}

**Created:** {item[7]}
"""

        # Check if files exist
        image_path = item[3] if item[3] and os.path.exists(item[3]) else None
        stl_path = item[4] if item[4] and os.path.exists(item[4]) else None
        glb_path = item[5] if item[5] and os.path.exists(item[5]) else None

        # Generate 3D viewer if GLB exists
        viewer_html = None
        viewer_visible = False
        if glb_path:
            viewer_html = create_3d_viewer_html(glb_path, container_height="500px")
            viewer_visible = True

        return (
            image_path,  # detail_image
            detail_text,  # detail_info
            stl_path if stl_path else gr.update(visible=False),  # detail_download_stl
            glb_path if glb_path else gr.update(visible=False),  # detail_download_glb
            gr.update(value=viewer_html, visible=True) if viewer_html else gr.update(visible=False),  # detail_viewer
            gr.update(visible=True, open=True)  # item_details_accordion
        )

    # Gallery event handlers
    demo.load(
        lambda: load_gallery_with_pagination(search="", items_to_show=20, view_mode="Grid") + (20,),
        inputs=None,
        outputs=[gallery_grid_view, gallery_grid_compact, gallery_status, items_shown_text, load_more_btn, grid_view, compact_view, items_to_show_state]
    )

    refresh_btn.click(
        lambda search, view_mode: load_gallery_with_pagination(search, items_to_show=20, view_mode=view_mode) + (20,),
        inputs=[search_input, view_mode],
        outputs=[gallery_grid_view, gallery_grid_compact, gallery_status, items_shown_text, load_more_btn, grid_view, compact_view, items_to_show_state]
    )

    search_input.submit(
        lambda search, view_mode: load_gallery_with_pagination(search, items_to_show=20, view_mode=view_mode) + (20,),
        inputs=[search_input, view_mode],
        outputs=[gallery_grid_view, gallery_grid_compact, gallery_status, items_shown_text, load_more_btn, grid_view, compact_view, items_to_show_state]
    )

    view_mode.change(
        handle_view_mode_change,
        inputs=[view_mode],
        outputs=[grid_view, compact_view]
    )

    # Gallery item selection for details
    gallery_grid_view.select(
        handle_gallery_item_select,
        inputs=[search_input],
        outputs=[detail_image, detail_info, detail_download_stl, detail_download_glb, detail_viewer, item_details_accordion]
    )

    # Load More button
    load_more_btn.click(
        handle_load_more,
        inputs=[search_input, view_mode, items_to_show_state],
        outputs=[gallery_grid_view, gallery_grid_compact, gallery_status, items_shown_text, load_more_btn, grid_view, compact_view, items_to_show_state]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        max_file_size="50mb"
    )
