"""
3D Viewer module for TRELLIS Streamlit application.

This module generates interactive Three.js-based 3D viewers for GLB/GLTF files.
The viewer supports features like rotation, zoom, wireframe mode, and auto-rotation.

Usage:
    from modules.three_d_viewer import create_3d_viewer_html, ThreeDViewer
    
    # Simple usage
    html = create_3d_viewer_html("model.glb")
    
    # Advanced usage with custom options
    viewer = ThreeDViewer(
        background_color="#ffffff",
        auto_rotate=True,
        show_controls=True
    )
    html = viewer.generate_html("model.glb")
"""

import base64
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ViewerConfig:
    """Configuration options for the 3D viewer."""
    
    # Appearance
    background_color: str = "#f5f5f5"
    container_height: str = "100vh"
    
    # Camera settings
    fov: int = 45
    near: float = 0.1
    far: int = 1000
    initial_position: tuple = (0, 0.5, 2)
    
    # Controls
    enable_damping: bool = True
    damping_factor: float = 0.05
    min_distance: float = 0.5
    max_distance: float = 10
    screen_space_panning: bool = True
    
    # Auto-rotation
    auto_rotate: bool = True
    auto_rotate_speed: float = 0.005
    
    # Model settings
    model_scale_factor: float = 1.5
    model_metalness: float = 0.3
    model_roughness: float = 0.7
    
    # UI Controls
    show_controls: bool = True
    show_info: bool = True
    show_wireframe_button: bool = True
    show_rotation_button: bool = True
    show_reset_button: bool = True
    
    # Lighting
    ambient_light_intensity: float = 1.0
    directional_lights: List[Dict[str, Any]] = None
    
    # Performance
    enable_antialias: bool = True
    tone_mapping: str = "ACESFilmicToneMapping"
    tone_mapping_exposure: float = 1.2
    
    def __post_init__(self):
        """Set default lighting if none provided."""
        if self.directional_lights is None:
            self.directional_lights = [
                {"color": "#ffffff", "intensity": 1.2, "position": (1, 1, 0.5)},
                {"color": "#ffffff", "intensity": 0.8, "position": (-1, 0.5, -0.5)},
                {"color": "#ffffff", "intensity": 0.5, "position": (0, -1, 0)}
            ]


class ThreeDViewer:
    """
    Advanced 3D viewer generator with configurable options.
    """
    
    def __init__(self, config: Optional[ViewerConfig] = None, normals: bool = False, **kwargs):
        """
        Initialize the 3D viewer.

        Args:
            config: ViewerConfig object with settings
            normals: When True, apply MeshNormalMaterial to every mesh instead of GLB materials
            **kwargs: Individual config options (override config object)
        """
        self.config = config or ViewerConfig()
        self.normals = normals

        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _load_glb_as_base64(self, glb_path: str) -> str:
        """Load GLB file and convert to base64."""
        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
        
        with open(glb_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def _generate_css(self) -> str:
        """Generate CSS styles for the viewer."""
        return f"""
            body {{ margin: 0; overflow: hidden; }}
            #container {{ width: 100%; height: {self.config.container_height}; }}
            #loader {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: Arial, sans-serif;
                font-size: 14px;
                color: #666;
            }}
            #controls {{
                position: absolute;
                bottom: 10px;
                left: 50%;
                transform: translateX(-50%);
                display: {'flex' if self.config.show_controls else 'none'};
                gap: 10px;
                background: rgba(255,255,255,0.9);
                padding: 8px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            button {{
                padding: 5px 10px;
                cursor: pointer;
                border: none;
                border-radius: 4px;
                background: #4CAF50;
                color: white;
                font-size: 12px;
                transition: background 0.3s;
            }}
            button:hover {{
                background: #45a049;
            }}
            button.active {{
                background: #2196F3;
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: #666;
                font-family: Arial, sans-serif;
                font-size: 11px;
                background: rgba(255,255,255,0.8);
                padding: 5px 8px;
                border-radius: 4px;
                display: {'block' if self.config.show_info else 'none'};
            }}
        """
    
    def _generate_controls_html(self) -> str:
        """Generate HTML for viewer controls."""
        if not self.config.show_controls:
            return ""
        
        buttons = []
        
        if self.config.show_reset_button:
            buttons.append('<button onclick="resetCamera()">🔄 Reset View</button>')
        
        if self.config.show_wireframe_button:
            buttons.append('<button onclick="toggleWireframe()" id="wireframeBtn">📐 Wireframe</button>')
        
        if self.config.show_rotation_button:
            active_class = "active" if self.config.auto_rotate else ""
            buttons.append(f'<button onclick="toggleRotation()" id="rotateBtn" class="{active_class}">🔄 Auto-Rotate</button>')
        
        return "".join(buttons)
    
    def _generate_lighting_js(self) -> str:
        """Generate JavaScript for scene lighting."""
        js_lines = [
            f"const ambientLight = new THREE.AmbientLight(0xffffff, {self.config.ambient_light_intensity});",
            "scene.add(ambientLight);"
        ]
        
        for i, light in enumerate(self.config.directional_lights):
            color = light.get("color", "#ffffff").replace("#", "0x")
            intensity = light.get("intensity", 1.0)
            pos = light.get("position", (1, 1, 1))
            
            js_lines.extend([
                f"const directionalLight{i+1} = new THREE.DirectionalLight({color}, {intensity});",
                f"directionalLight{i+1}.position.set({pos[0]}, {pos[1]}, {pos[2]});",
                f"scene.add(directionalLight{i+1});"
            ])
        
        return "\n            ".join(js_lines)
    
    def _generate_material_js(self) -> str:
        """Generate JavaScript for per-mesh material setup inside traverse."""
        if self.normals:
            return "child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });"
        return (
            "child.material.side = THREE.DoubleSide;\n"
            "                            // Ensure materials are well-lit\n"
            "                            if (child.material.metalness !== undefined) {{\n"
            f"                                child.material.metalness = {self.config.model_metalness};\n"
            f"                                child.material.roughness = {self.config.model_roughness};\n"
            "                            }}"
        )

    def _generate_javascript(self, glb_data: str) -> str:
        """Generate JavaScript code for the 3D viewer."""
        return f"""
            let scene, camera, renderer, controls, model, mixer;
            let autoRotate = {str(self.config.auto_rotate).lower()};
            let wireframeMode = false;
            const container = document.getElementById('container');
            
            // Initialize scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x{self.config.background_color.replace('#', '')});
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(
                {self.config.fov},
                window.innerWidth / window.innerHeight,
                {self.config.near},
                {self.config.far}
            );
            camera.position.set({self.config.initial_position[0]}, {self.config.initial_position[1]}, {self.config.initial_position[2]});
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({{ antialias: {str(self.config.enable_antialias).lower()} }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.toneMapping = THREE.{self.config.tone_mapping};
            renderer.toneMappingExposure = {self.config.tone_mapping_exposure};
            container.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = {str(self.config.enable_damping).lower()};
            controls.dampingFactor = {self.config.damping_factor};
            controls.screenSpacePanning = {str(self.config.screen_space_panning).lower()};
            controls.minDistance = {self.config.min_distance};
            controls.maxDistance = {self.config.max_distance};
            controls.maxPolarAngle = Math.PI;
            
            // Lights
            {self._generate_lighting_js()}
            
            // Load GLB model from base64
            const loader = new THREE.GLTFLoader();
            const glbData = "{glb_data}";
            const glbBinary = Uint8Array.from(atob(glbData), c => c.charCodeAt(0));
            const glbBlob = new Blob([glbBinary], {{ type: 'model/gltf-binary' }});
            const glbUrl = URL.createObjectURL(glbBlob);
            
            loader.load(
                glbUrl,
                function (gltf) {{
                    model = gltf.scene;
                    
                    // Center and scale the model
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = {self.config.model_scale_factor} / maxDim;
                    
                    model.scale.multiplyScalar(scale);
                    model.position.sub(center.multiplyScalar(scale));
                    model.position.y = 0;
                    
                    // Setup materials
                    model.traverse((child) => {{
                        if (child.isMesh) {{
                            {self._generate_material_js()}
                        }}
                    }});
                    
                    scene.add(model);
                    
                    // Setup animation mixer if model has animations
                    if (gltf.animations.length > 0) {{
                        mixer = new THREE.AnimationMixer(model);
                        gltf.animations.forEach((clip) => {{
                            mixer.clipAction(clip).play();
                        }});
                    }}
                    
                    // Hide loader
                    document.getElementById('loader').style.display = 'none';
                    
                    // Set optimal camera position
                    const distance = maxDim * 2;
                    camera.position.set(distance * 0.5, distance * 0.5, distance);
                    camera.lookAt(0, 0, 0);
                    controls.target.set(0, 0, 0);
                    controls.update();
                }},
                function (xhr) {{
                    // Progress callback
                    const percentComplete = (xhr.loaded / xhr.total) * 100;
                    document.getElementById('loader').textContent = 
                        'Loading 3D model... ' + Math.round(percentComplete) + '%';
                }},
                function (error) {{
                    console.error('Error loading model:', error);
                    document.getElementById('loader').textContent = 'Error loading model';
                }}
            );
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                
                if (autoRotate && model) {{
                    model.rotation.y += {self.config.auto_rotate_speed};
                }}
                
                if (mixer) {{
                    mixer.update(0.016);
                }}
                
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Window resize handler
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
            
            // Control functions
            function resetCamera() {{
                camera.position.set({self.config.initial_position[0]}, {self.config.initial_position[1]}, {self.config.initial_position[2]});
                controls.target.set(0, 0, 0);
                controls.update();
            }}
            
            function toggleWireframe() {{
                wireframeMode = !wireframeMode;
                if (document.getElementById('wireframeBtn')) {{
                    document.getElementById('wireframeBtn').classList.toggle('active', wireframeMode);
                }}
                if (model) {{
                    model.traverse((child) => {{
                        if (child.isMesh) {{
                            child.material.wireframe = wireframeMode;
                        }}
                    }});
                }}
            }}
            
            function toggleRotation() {{
                autoRotate = !autoRotate;
                if (document.getElementById('rotateBtn')) {{
                    document.getElementById('rotateBtn').classList.toggle('active', autoRotate);
                }}
            }}
        """
    
    def generate_html(self, glb_path: str) -> str:
        """
        Generate complete HTML for the 3D viewer.
        
        Args:
            glb_path: Path to the GLB file to display
            
        Returns:
            str: Complete HTML string for the 3D viewer
        """
        glb_data = self._load_glb_as_base64(glb_path)
        
        html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
        {self._generate_css()}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <div id="loader">Loading 3D model...</div>
        <div id="info">🖱️ Drag to rotate • Scroll to zoom • Right-click to pan</div>
        <div id="controls">
            {self._generate_controls_html()}
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        
        <script>
        {self._generate_javascript(glb_data)}
        </script>
    </body>
    </html>
        """
        
        return html_string.strip()


# Convenience function for simple usage (backward compatibility)
def create_3d_viewer_html(glb_path: str, normals: bool = False, **options) -> str:
    """
    Create HTML for 3D viewer with default settings.

    Args:
        glb_path: Path to GLB file to display
        normals: When True, apply MeshNormalMaterial to every mesh instead of GLB materials
        **options: Optional configuration overrides

    Returns:
        str: Complete HTML string for the 3D viewer
    """
    viewer = ThreeDViewer(normals=normals, **options)
    return viewer.generate_html(glb_path)


# Preset configurations
def create_minimal_viewer(glb_path: str) -> str:
    """Create a minimal 3D viewer with no controls."""
    config = ViewerConfig(
        show_controls=False,
        show_info=False,
        auto_rotate=True
    )
    viewer = ThreeDViewer(config)
    return viewer.generate_html(glb_path)


def create_showcase_viewer(glb_path: str) -> str:
    """Create a showcase viewer optimized for presentations."""
    config = ViewerConfig(
        background_color="#1a1a1a",
        auto_rotate=True,
        auto_rotate_speed=0.003,
        ambient_light_intensity=0.8,
        tone_mapping_exposure=1.5,
        show_wireframe_button=False
    )
    viewer = ThreeDViewer(config)
    return viewer.generate_html(glb_path)


def create_technical_viewer(glb_path: str) -> str:
    """Create a technical viewer with wireframe and detailed controls."""
    config = ViewerConfig(
        background_color="#ffffff",
        auto_rotate=False,
        show_wireframe_button=True,
        model_metalness=0.1,
        model_roughness=0.9,
        ambient_light_intensity=1.2
    )
    viewer = ThreeDViewer(config)
    return viewer.generate_html(glb_path)


# Global viewer instance for caching
_default_viewer = None


def get_default_viewer() -> ThreeDViewer:
    """Get a cached default viewer instance."""
    global _default_viewer
    if _default_viewer is None:
        _default_viewer = ThreeDViewer()
    return _default_viewer