import os
import streamlit as st

# App title and description - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Text-to-3D Generator", page_icon="🎨", layout="wide")

import streamlit.components.v1 as components
import torch
import gc
from PIL import Image
from modules.simple_stl_converter import convert_glb_to_stl
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

# Initialize session state for storing models and generated content
if 'flux_pipe' not in st.session_state:
    st.session_state.flux_pipe = None
if 'trellis_pipeline' not in st.session_state:
    st.session_state.trellis_pipeline = None
if 'reward_model' not in st.session_state:
    st.session_state.reward_model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'selected_image_index' not in st.session_state:
    st.session_state.selected_image_index = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'glb_path' not in st.session_state:
    st.session_state.glb_path = None
if 'using_meshfleet' not in st.session_state:
    st.session_state.using_meshfleet = False
if 'stl_file_path' not in st.session_state:
    st.session_state.stl_file_path = None
if 'stl_file_number' not in st.session_state:
    st.session_state.stl_file_number = None
if 'stl_prepared' not in st.session_state:
    st.session_state.stl_prepared = False
if 'guidance_scale' not in st.session_state:
    st.session_state.guidance_scale = 4.5
if 'num_inference_steps' not in st.session_state:
    st.session_state.num_inference_steps = 28
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'generation_step' not in st.session_state:
    st.session_state.generation_step = 0  # 0: initial, 1: images generated, 2: 3D generated
if 'gallery_view' not in st.session_state:
    st.session_state.gallery_view = 'grid'  # 'grid' or 'list'


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
    # Default: both on cuda:0 with CPU offloading for FLUX
    return None

def get_trellis_cpu_offload_setting():
    """Check if TRELLIS CPU offloading is requested."""
    return os.environ.get("ENABLE_TRELLIS_CPU_OFFLOAD") == "true"

def get_gaussian_rendering_setting():
    """Check if Gaussian rendering is requested."""
    return os.environ.get("USE_GAUSSIAN_RENDERING") == "true"

# Gallery manager initialization (cached)
@st.cache_resource
def load_gallery_manager():
    """Load gallery management system once and cache it."""
    return get_gallery_manager()

# Initialize session state for gallery manager
if 'gallery_manager' not in st.session_state:
    st.session_state.gallery_manager = None

# Content moderation initialization (cached)
@st.cache_resource
def load_content_moderator():
    """Load content moderation system once and cache it."""
    return get_content_moderator()

# Initialize session state for content moderator
if 'content_moderator' not in st.session_state:
    st.session_state.content_moderator = None

# Model manager initialization (cached)
@st.cache_resource
def load_model_manager():
    """Load model management system once and cache it."""
    device_config = get_device_config()
    enable_trellis_cpu_offload = get_trellis_cpu_offload_setting()
    image_model = os.environ.get('IMAGE_MODEL', 'flux')

    if enable_trellis_cpu_offload:
        st.info("🔄 TRELLIS CPU offloading enabled - this will reduce VRAM usage but may be slower")

    return get_model_manager(device_config, enable_trellis_cpu_offload, image_model=image_model)

# Initialize session state for model manager
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None

# Generation pipeline initialization (cached)
@st.cache_resource
def load_generation_pipeline():
    """Load generation pipeline system once and cache it."""
    return get_generation_pipeline()

# Initialize session state for generation pipeline
if 'generation_pipeline' not in st.session_state:
    st.session_state.generation_pipeline = None

if 'use_gaussian_rendering' not in st.session_state:
    st.session_state.use_gaussian_rendering = get_gaussian_rendering_setting()
if 'is_generating_images' not in st.session_state:
    st.session_state.is_generating_images = False
if 'is_generating_3d' not in st.session_state:
    st.session_state.is_generating_3d = False

# Function to reset only the generated content (preserve models)
def reset_generated_content():
    """Reset only generated content, preserving loaded models"""
    st.session_state.generated_images = []
    st.session_state.selected_image_index = None
    st.session_state.video_path = None
    st.session_state.glb_path = None
    st.session_state.stl_file_path = None
    st.session_state.stl_file_number = None
    st.session_state.stl_prepared = False
    st.session_state.current_prompt = ""
    st.session_state.generation_step = 0
    st.session_state.is_generating_images = False
    st.session_state.is_generating_3d = False
    # Clear memory but keep models
    torch.cuda.empty_cache()
    gc.collect()

# Function to load models using ModelManager
@st.cache_resource
def load_models():
    """Load models using the ModelManager."""
    # Ensure model manager is loaded
    if st.session_state.model_manager is None:
        st.session_state.model_manager = load_model_manager()
    
    # Load all models
    flux_pipe, trellis_pipe, reward_model = st.session_state.model_manager.load_all_models()
    
    # Update session state with generation config
    config = st.session_state.model_manager.get_generation_config()
    st.session_state.guidance_scale = config["guidance_scale"]
    st.session_state.num_inference_steps = config["num_inference_steps"]
    
    return flux_pipe, trellis_pipe, reward_model


# Function to generate images using GenerationPipeline
def generate_images(prompt, num_images=4, base_seed=None):
    # Ensure generation pipeline is loaded and configured
    if st.session_state.generation_pipeline is None:
        st.session_state.generation_pipeline = load_generation_pipeline()
    
    # Ensure content moderator is loaded
    if st.session_state.content_moderator is None:
        st.session_state.content_moderator = load_content_moderator()
    
    # Set pipeline models
    st.session_state.generation_pipeline.set_models(
        flux_pipeline=st.session_state.flux_pipe,
        trellis_pipeline=st.session_state.trellis_pipeline,
        reward_model=st.session_state.reward_model,
        content_moderator=st.session_state.content_moderator
    )
    
    with st.spinner(f"🎨 Creating images from your description..."):
        filtered_images = st.session_state.generation_pipeline.generate_images(
            prompt, 
            num_images=num_images, 
            base_seed=base_seed,
            guidance_scale=st.session_state.guidance_scale,
            num_inference_steps=st.session_state.num_inference_steps
        )
        
        st.session_state.generated_images = filtered_images
    return filtered_images

# Function to convert selected image to 3D using GenerationPipeline
def generate_3d_model(image, base_seed=None):
    # Ensure generation pipeline is loaded and configured
    if st.session_state.generation_pipeline is None:
        st.session_state.generation_pipeline = load_generation_pipeline()
    
    # Set pipeline models
    st.session_state.generation_pipeline.set_models(
        flux_pipeline=st.session_state.flux_pipe,
        trellis_pipeline=st.session_state.trellis_pipeline,
        reward_model=st.session_state.reward_model,
        content_moderator=st.session_state.content_moderator
    )

    sample_video = st.session_state.use_gaussian_rendering
    use_simple_glb = not st.session_state.use_gaussian_rendering

    with st.spinner(f"🔮 Transforming image into 3D model..."):
        video_path, glb_path = st.session_state.generation_pipeline.generate_3d_model(
            image, 
            base_seed=base_seed, 
            sample_video=sample_video,
            use_simple_glb=use_simple_glb
        )
        return video_path, glb_path

# def prepare_3d_model_for_printing():
#     if st.session_state.stl_prepared:
#         with open(st.session_state.stl_file_path, "rb") as file:
#             return file.read()
    
#     # Ensure gallery manager is loaded
#     if st.session_state.gallery_manager is None:
#         st.session_state.gallery_manager = load_gallery_manager()
    
#     # Get next file number and increment counter
#     file_number = st.session_state.gallery_manager.increment_counter()
#     output_dir = st.session_state.gallery_manager.output_dir

#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     glb_output = os.path.join(output_dir, f"model_{file_number}.glb")
#     # Save the GLB file
#     with open(glb_output, "wb") as f:
#         with open(st.session_state.glb_path, "rb") as f_in:
#             f.write(f_in.read())
    
#     stl_filepath = print_pipeline.run_with_file(glb_output, file_number, output_folder=output_dir)
    
#     # Store the STL file path in session state for download
#     st.session_state.stl_file_path = stl_filepath
#     st.session_state.stl_file_number = file_number
#     st.session_state.stl_prepared = True
    
#     # Save to gallery
#     if st.session_state.selected_image_index is not None and st.session_state.generated_images:
#         selected_image = st.session_state.generated_images[st.session_state.selected_image_index]
#         st.session_state.gallery_manager.save_item(
#             file_number,
#             st.session_state.current_prompt,
#             selected_image,
#             stl_filepath,
#             st.session_state.glb_path,
#             st.session_state.video_path
#         )

#     with open(st.session_state.stl_file_path, "rb") as file:
#         return file.read()

def prepare_3d_model_for_printing():
    if st.session_state.stl_prepared:
        with open(st.session_state.stl_file_path, "rb") as file:
            return file.read()
    
    # Ensure gallery manager is loaded
    if st.session_state.gallery_manager is None:
        st.session_state.gallery_manager = load_gallery_manager()
    
    # Get next file number and increment counter
    file_number = st.session_state.gallery_manager.increment_counter()
    output_dir = st.session_state.gallery_manager.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    glb_output = os.path.join(output_dir, f"model_{file_number}.glb")
    # Save the GLB file
    with open(glb_output, "wb") as f:
        with open(st.session_state.glb_path, "rb") as f_in:
            f.write(f_in.read())

    # Generate STL (simple conversion without base plate)
    stl_filepath = convert_glb_to_stl(glb_output, file_number, output_folder=output_dir)
    
    # Store the STL file path in session state for download
    st.session_state.stl_file_path = stl_filepath
    st.session_state.stl_file_number = file_number
    st.session_state.stl_prepared = True
    
    # Save to gallery
    if st.session_state.selected_image_index is not None and st.session_state.generated_images:
        selected_image = st.session_state.generated_images[st.session_state.selected_image_index]
        st.session_state.gallery_manager.save_item(
            file_number,
            st.session_state.current_prompt,
            selected_image,
            stl_filepath,
            st.session_state.glb_path,
            st.session_state.video_path
        )

    with open(st.session_state.stl_file_path, "rb") as file:
        return file.read()

def display_gallery():
    """Display the gallery section"""
    st.markdown("## 🖼️ Gallery & History")
    
    # Ensure gallery manager is loaded
    if st.session_state.gallery_manager is None:
        st.session_state.gallery_manager = load_gallery_manager()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search by print number or prompt", placeholder="e.g., 0001 or dragon")
    with col2:
        view_mode = st.radio("View", ["Grid", "List"], horizontal=True)
    with col3:
        if st.button("🔄 Refresh Gallery"):
            st.rerun()
    
    gallery_items = st.session_state.gallery_manager.load_items(search_query=search_query if search_query else None)
    
    if not gallery_items:
        st.info("No items in gallery yet. Start creating to see them here!")
        return
    
    if view_mode == "Grid":
        # Grid view
        cols_per_row = 4
        for i in range(0, len(gallery_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(gallery_items):
                    item = gallery_items[i + j]
                    with col:
                        # Display thumbnail
                        if os.path.exists(item[3]):  # image_path
                            image = Image.open(item[3])
                            st.image(image, use_container_width=True)
                        
                        st.markdown(f"**#{item[1]}**")  # print_number
                        st.caption(item[2][:50] + "..." if len(item[2]) > 50 else item[2])  # prompt
                        
                        # View details button
                        if st.button(f"View", key=f"view_{item[0]}", use_container_width=True):
                            with st.expander(f"Details for #{item[1]}", expanded=True):
                                col_img, col_info = st.columns(2)
                                with col_img:
                                    if os.path.exists(item[3]):
                                        st.image(Image.open(item[3]), use_container_width=True)
                                with col_info:
                                    st.markdown(f"**Print Number:** #{item[1]}")
                                    st.markdown(f"**Prompt:** {item[2]}")
                                    st.markdown(f"**Created:** {item[7]}")
                                    
                                    # Download buttons
                                    if item[4] and os.path.exists(item[4]):  # STL path
                                        with open(item[4], "rb") as f:
                                            st.download_button(
                                                "📥 Download STL",
                                                f.read(),
                                                file_name=f"model_{item[1]}.stl",
                                                mime="application/octet-stream",
                                                use_container_width=True
                                            )
                                    
                                    if item[5] and os.path.exists(item[5]):  # GLB path
                                        with open(item[5], "rb") as f:
                                            st.download_button(
                                                "📥 Download GLB",
                                                f.read(),
                                                file_name=f"model_{item[1]}.glb",
                                                mime="application/octet-stream",
                                                use_container_width=True
                                            )
                                
                                # 3D Viewer
                                if item[5] and os.path.exists(item[5]):
                                    st.markdown("### 🎮 Interactive 3D View")
                                    viewer_html = create_3d_viewer_html(item[5])
                                    components.html(viewer_html, height=400)
    else:
        # List view
        for item in gallery_items:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
                
                with col1:
                    if os.path.exists(item[3]):  # image_path
                        st.image(Image.open(item[3]), width=100)
                
                with col2:
                    st.markdown(f"**#{item[1]}** - {item[2]}")
                    st.caption(f"Created: {item[7]}")
                
                with col3:
                    if item[5] and os.path.exists(item[5]):
                        if st.button("🎮 3D", key=f"3d_{item[0]}"):
                            st.session_state[f"show_3d_{item[0]}"] = not st.session_state.get(f"show_3d_{item[0]}", False)
                
                with col4:
                    if item[4] and os.path.exists(item[4]):  # STL path
                        with open(item[4], "rb") as f:
                            st.download_button(
                                "STL",
                                f.read(),
                                file_name=f"model_{item[1]}.stl",
                                mime="application/octet-stream"
                            )
                
                with col5:
                    if item[5] and os.path.exists(item[5]):  # GLB path
                        with open(item[5], "rb") as f:
                            st.download_button(
                                "GLB",
                                f.read(),
                                file_name=f"model_{item[1]}.glb",
                                mime="application/octet-stream"
                            )
                
                # Show 3D viewer if toggled
                if st.session_state.get(f"show_3d_{item[0]}", False) and item[5] and os.path.exists(item[5]):
                    viewer_html = create_3d_viewer_html(item[5])
                    components.html(viewer_html, height=400)
                
                st.divider()

# SIDEBAR
with st.sidebar:
    st.title("🎮 Control Panel")
    
    # Initialize seed variables with defaults
    image_base_seed = 0
    model_base_seed = 0
    
    # Status indicator
    st.markdown("### 📊 Status")

    # Show image model type
    image_model_type = os.environ.get('IMAGE_MODEL', 'flux').upper()
    st.info(f"📷 Image Model: **{image_model_type}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.flux_pipe is None:
            st.info("🔄 Models not loaded")
        else:
            # Show device configuration for FLUX, API status for Gemini
            if st.session_state.model_manager:
                if image_model_type == "FLUX":
                    device_config = st.session_state.model_manager.device_config
                    if device_config.get("flux") != device_config.get("trellis"):
                        st.success("✅ Multi-GPU")
                    else:
                        st.success("✅ Models ready")
                else:
                    st.success("✅ API ready")
            else:
                st.success("✅ Models ready")
    with col2:
        # Load content moderator if not already loaded
        if st.session_state.content_moderator is None:
            st.session_state.content_moderator = load_content_moderator()
        
        status_items = st.session_state.content_moderator.get_safety_summary()
        
        if status_items:
            st.success(f"🛡️ Safety: {', '.join(status_items)}")
        else:
            st.warning("⚠️ No safety checks")

    # Optional features status
    if st.session_state.model_manager and st.session_state.flux_pipe is not None:
        features = []
        if st.session_state.reward_model is not None:
            features.append("🎯 Quality Ranking")
        if st.session_state.model_manager.enable_trellis_cpu_offload:
            features.append("💾 CPU Offload")
        if features:
            st.caption(f"Active: {' · '.join(features)}")
        else:
            st.caption("ℹ️ No optional features active")

    # Next file number display
    st.markdown("### 📦 Next Print Number")
    # Load gallery manager if needed
    if st.session_state.gallery_manager is None:
        st.session_state.gallery_manager = load_gallery_manager()
    next_number = st.session_state.gallery_manager.get_next_print_number()
    st.info(f"**#{next_number}**")
    
    # Progress tracker
    st.markdown("### 📈 Progress")
    if st.session_state.generation_step == 0:
        st.progress(0.0, "Ready to start")
    elif st.session_state.generation_step == 1:
        st.progress(0.5, "Images generated")
    elif st.session_state.generation_step == 2:
        st.progress(1.0, "3D model ready!")
    
    
    # Advanced settings (only show when models are loaded)
    if st.session_state.flux_pipe is not None:
        st.markdown("### ⚙️ Advanced Settings")
        
        with st.expander("🎲 Seed Settings", expanded=False):
            st.info("Seeds control randomness. Same seed = same result. Leave at 0 for random generation each time.")
            col1, col2 = st.columns(2)
            with col1:
                image_base_seed = st.number_input("Image Base Seed", value=0, min_value=0, max_value=999999, 
                                                help="Set to 0 for random, or use a specific number for reproducible results")
            with col2:
                model_base_seed = st.number_input("3D Base Seed", value=0, min_value=0, max_value=999999,
                                                help="Set to 0 for random, or use a specific number for reproducible results")
        
        num_images = st.slider("Images to Generate", min_value=2, max_value=8, value=4,
                               help="More images = more options but takes longer")
    
    # Action buttons
    st.markdown("### 🎯 Actions")
    
    if st.button("🔄 Start New Project", use_container_width=True,
                 disabled=(st.session_state.generation_step == 0),
                 help="Reset everything except the loaded models"):
        reset_generated_content()
        st.rerun()
    
    # Gallery stats
    st.markdown("### 📊 Gallery Stats")
    # Load gallery manager if needed  
    if st.session_state.gallery_manager is None:
        st.session_state.gallery_manager = load_gallery_manager()
    stats = st.session_state.gallery_manager.get_stats()
    st.metric("Total Creations", stats["total_items"])
    
    # History section
    if st.session_state.stl_file_number:
        st.markdown("### 📜 Recent")
        st.success(f"Last saved: **#{st.session_state.stl_file_number}**")

# MAIN CONTENT
st.title("🎨 Text-to-3D Print Generator")
st.markdown("Transform your ideas into printable 3D objects using AI")

# Create tabs
tab1, tab2 = st.tabs(["🚀 Create New", "🖼️ Gallery"])

with tab1:
    # Load models if not loaded
    if st.session_state.flux_pipe is None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🚀 Initialize AI Models", use_container_width=True, type="primary"):
                # Ensure model manager is loaded
                if st.session_state.model_manager is None:
                    st.session_state.model_manager = load_model_manager()
                
                flux_pipe, trellis_pipeline, reward_model = load_models()
                st.session_state.reward_model = reward_model
                st.session_state.flux_pipe = flux_pipe
                st.session_state.trellis_pipeline = trellis_pipeline
                st.success("✅ Models loaded successfully! You can now start creating.")
                st.rerun()
        
        st.info("""
        **First time here?** Click the button above to load the AI models. 
        This one-time process takes 1-2 minutes but enables instant generation afterwards.
        """)
        
    else:
        # Main workflow
        # Step indicator
        steps = ["📝 Describe", "🖼️ Select", "🎯 Generate 3D"]
        step_cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(step_cols, steps)):
            with col:
                if i <= st.session_state.generation_step:
                    st.success(step)
                else:
                    st.info(step)
        
        st.markdown("---")
        
        # STEP 1: Text input
        st.markdown("### Step 1: Describe Your Object")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Example prompts with categories
            example_categories = {
                "🦄 Fantasy Creatures": [
                    "A Goblin riding a Roomba vacuum cleaner into battle",
                    "A dragon curled around a treasure chest, sleeping peacefully",
                    "A unicorn with mechanical wings"
                ],
                "🐾 Animals": [
                    "A Capybara knight in full plate armor, looking serene and noble",
                    "A Badger monk in a meditative pose",
                    "A T-Rex wearing a tiny top hat and a monocle",
                    "A penguin astronaut with a jetpack"
                ],
                "🎮 Gaming": [
                    "A dice with elevated mystical runes on each face",
                    "A chess piece that's a hybrid of knight and dragon",
                    "A game controller transformed into a robot"
                ],
                "🏠 Everyday Objects": [
                    "A coffee mug shaped like a sleeping cat",
                    "A phone holder designed as tiny hands",
                    "A succulent planter shaped like a skull"
                ]
            }
            
            selected_category = st.selectbox("💡 Get inspired:", ["Custom"] + list(example_categories.keys()))
            
            if selected_category != "Custom":
                selected_example = st.selectbox("Choose an example:", example_categories[selected_category])
            else:
                selected_example = ""
        
        with col2:
            st.markdown("#### Tips for best results:")
            st.markdown("""
            - Be specific and detailed
            - Include the pose/position
            - Mention key features
            - Think "printable" and avoid fragile structures
            """)
            
            # Add moderation notice  
            if st.session_state.content_moderator and st.session_state.content_moderator.text_moderation_enabled:
                st.caption("🛡️ Content moderation active")
        
        prompt = st.text_area("✨ Enter your object description:", 
                              height=100, 
                              value=selected_example,
                              placeholder="E.g., 'A wise owl wearing glasses, holding a tiny book'",
                              key="prompt_input")
        
        # Function to handle generate button click
        def handle_generate_click():
            # Store the prompt and variables needed for generation
            st.session_state.pending_prompt = prompt
            st.session_state.pending_image_seed = image_base_seed if 'image_base_seed' in locals() and image_base_seed > 0 else None
            st.session_state.pending_num_images = num_images if 'num_images' in locals() else 4
            st.session_state.is_generating_images = True
        
        # Generate images button with moderation
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # Dynamic button text based on whether images exist
            button_text = "🎨 Generate New Images" if st.session_state.generated_images else "🎨 Generate Images"
            st.button(button_text, use_container_width=True, type="primary", 
                     disabled=st.session_state.is_generating_images or st.session_state.is_generating_3d,
                     on_click=handle_generate_click)
        
        # Handle the actual generation if button was clicked
        if st.session_state.is_generating_images and hasattr(st.session_state, 'pending_prompt'):
            # Ensure content moderator is loaded
            if st.session_state.content_moderator is None:
                st.session_state.content_moderator = load_content_moderator()
            
            # Check content moderation first
            is_safe, scores = st.session_state.content_moderator.check_text_safety(st.session_state.pending_prompt)
            
            if is_safe:
                # Proceed with generation
                st.session_state.generated_images = generate_images(
                    st.session_state.pending_prompt, 
                    num_images=st.session_state.pending_num_images, 
                    base_seed=st.session_state.pending_image_seed
                )
                st.session_state.selected_image_index = None
                st.session_state.video_path = None
                st.session_state.glb_path = None
                st.session_state.current_prompt = st.session_state.pending_prompt
                st.session_state.generation_step = 1
                
                # Clean up and reset state
                st.session_state.is_generating_images = False
                del st.session_state.pending_prompt
                del st.session_state.pending_image_seed
                del st.session_state.pending_num_images
                st.rerun()
            else:
                # Show error and reset state
                st.error("⚠️ Your prompt was flagged for potentially inappropriate content.")
                st.warning("Please modify your description to focus on printable objects.")
                st.session_state.is_generating_images = False
                del st.session_state.pending_prompt
                del st.session_state.pending_image_seed 
                del st.session_state.pending_num_images
                    
        
        # Function to handle image selection button click
        def handle_image_selection(image_index):
            # Store the selection parameters for processing
            st.session_state.pending_image_index = image_index
            st.session_state.pending_model_seed = model_base_seed if 'model_base_seed' in locals() and model_base_seed > 0 else None
            st.session_state.is_generating_3d = True
            st.session_state.selected_image_index = image_index
        
        # STEP 2: Display generated images
        if st.session_state.generated_images:
            st.markdown("---")
            st.markdown("### Step 2: Choose Your Favorite")
            
            st.info("💡 Click on the button below the image you like best")
            
            # Display images in a grid
            cols = st.columns(len(st.session_state.generated_images))
            for i, (col, img) in enumerate(zip(cols, st.session_state.generated_images)):
                with col:
                    st.image(img, use_container_width=True, output_format="PNG")

                    st.button(f"Choose #{i+1}", key=f"select_{i}", use_container_width=True,
                             type="primary" if st.session_state.selected_image_index == i else "secondary",
                             disabled=st.session_state.is_generating_3d,
                             on_click=handle_image_selection, args=(i,))
        
        # Handle 3D generation if image was selected
        if st.session_state.is_generating_3d and hasattr(st.session_state, 'pending_image_index'):
            # Automatically generate 3D model
            selected_image = st.session_state.generated_images[st.session_state.pending_image_index]
            
            st.session_state.video_path, st.session_state.glb_path = generate_3d_model(
                selected_image, 
                base_seed=st.session_state.pending_model_seed
            )
            st.session_state.generation_step = 2
            st.session_state.stl_prepared = False
            
            # Clean up and reset state
            st.session_state.is_generating_3d = False
            del st.session_state.pending_image_index
            del st.session_state.pending_model_seed
            st.rerun()
        
        # STEP 3: Display 3D results
        if st.session_state.glb_path:
            st.markdown("---")
            st.markdown("### Step 3: Your 3D Model is Ready! 🎉")
        
            # check if video path is empty, if so only show 3D viewer
            if len(st.session_state.video_path) == 0:
                st.markdown("#### 🎮 Interactive 3D Model")
                viewer_html = create_3d_viewer_html(st.session_state.glb_path)
                components.html(viewer_html, height=550)
            else:
                # Display video and 3D viewer side by side - adjusted proportions
                col1, col2 = st.columns([1, 2])  # Changed from [2, 2] to [1, 2] to make video column smaller
                with col1:
                    st.markdown("#### 🎬 Gaussian Splatting")
                    st.video(st.session_state.video_path)
                    st.caption("*AI's interpretation of the 3D structure*")
                
                with col2:
                    st.markdown("#### 🎮 Interactive 3D Model")
                    viewer_html = create_3d_viewer_html(st.session_state.glb_path)
                    components.html(viewer_html, height=550)  # Increased from 400 to 550
                
            st.markdown("---")
            
            # Download section
            st.markdown("### 📥 Download Options")

            col1, col2 = st.columns(2)

            with col1:
                # Show print number that will be assigned
                if not st.session_state.stl_prepared:
                    # Load gallery manager if needed
                    if st.session_state.gallery_manager is None:
                        st.session_state.gallery_manager = load_gallery_manager()
                    next_number = st.session_state.gallery_manager.get_next_print_number()
                    st.info(f"🏷️ Print #: **{next_number}**")
                else:
                    st.success(f"✅ Saved as **#{st.session_state.stl_file_number}**")
            
            with col2:
                # Download STL button
                stl_data = prepare_3d_model_for_printing()
                # st.download_button(
                #     label="📦 Prepare for Print",
                #     data=stl_data,
                #     file_name=f"model_{st.session_state.stl_file_number}.stl",
                #     mime="application/octet-stream",
                #     use_container_width=True,
                #     type="primary"
                # )
            
            # Quick actions
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("✨ Create Another Object", use_container_width=True, type="primary"):
                    reset_generated_content()
                    st.rerun()
            
            # Additional info
            with st.expander("ℹ️ About your files"):
                st.markdown(f"""
                **File**: Ready for 3D printing with a {30}mm diameter base and your unique number #{st.session_state.stl_file_number if st.session_state.stl_file_number else st.session_state.gallery_manager.get_next_print_number() if st.session_state.gallery_manager else '0001'} engraved.
                
                **Print Collection**: Use your print number to identify your object when collecting!
                
                **3D Viewer Controls**:
                - 🖱️ **Left Click + Drag**: Rotate the model
                - 🖱️ **Scroll**: Zoom in/out
                - 🖱️ **Right Click + Drag**: Pan the view
                - 📐 **Wireframe**: Toggle to see the mesh structure
                - 🔄 **Auto-Rotate**: Automatic rotation for showcase
                """)

with tab2:
    display_gallery()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Powered by FLUX and TRELLIS 
</div>
""", unsafe_allow_html=True)