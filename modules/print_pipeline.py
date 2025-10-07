import trimesh
import numpy as np
import subprocess

# ----------------------------- Konfiguration -----------------------------

TARGET_RADIUS = 15            # mm (30 mm Durchmesser)
BASE_THICKNESS = 0.8            # mm
FILE_NUMBER = "0001"
TEXT_HEIGHT = 0.4             # Gravurtiefe
TEXT_MARGIN = 5               # mm Abstand vom Rand

# Dateinamen
BASE_STL = "base.stl"
TEXT_SCAD = "text_model.scad"
TEXT_STL = "text_model.stl"
ENGRAVED_STL = "base_engraved.stl"
MODEL_PATH = "test_2.glb"
OUTPUT_STL = FILE_NUMBER+".stl"

# ------------------------- Baseplate erzeugen ----------------------------

def create_baseplate(radius, thickness, output_path):
    base = trimesh.creation.cylinder(radius=radius, height=thickness, sections=64)
    base.apply_translation([0, 0, thickness / 2])
    base.export(output_path)

# ------------------------ Text extrudieren (OpenSCAD) --------------------

def generate_text_stl(text, height, scad_path, stl_path):
    scad_code = f"""
    linear_extrude(height={height})
    text("{text}", size=20, font="Liberation Sans");
    """
    with open(scad_path, "w") as f:
        f.write(scad_code)

    subprocess.run(["openscad", "-o", stl_path, scad_path], check=True)
    print(f"✅ Text-STL erzeugt: {stl_path}")

# --------------------- Boolesche Differenz (OpenSCAD) --------------------

def scad_boolean_difference(base_path, subtract_path, output_path):
    scad_code = f"""
    difference() {{
        import("{base_path}");
        import("{subtract_path}");
    }}
    """
    with open("boolean_temp.scad", "w") as f:
        f.write(scad_code)

    subprocess.run(["openscad", "-o", output_path, "boolean_temp.scad"], check=True)
    print(f"✅ Gravierte Base erzeugt: {output_path}")

# ------------------------Slicing with Prusa Slicer -----------------------

def slice_with_prusaslicer(
    stl_file: str,
    OUTPUT_STL: str,
    config_file: str,
    prusaslicer_path: str = "PrusaSlicer"  # oder vollständiger Pfad zur .exe
):
    cmd = [
        prusaslicer_path,
        "--load", config_file,
        "--slice",
        "--output", OUTPUT_STL,
        stl_file
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ G-Code erzeugt: {OUTPUT_STL}")

# ----------------------------- Main Pipeline -----------------------------
def run_with_file(glb_path, file_number, output_folder="output"):
    global FILE_NUMBER, MODEL_PATH, OUTPUT_STL

    FILE_NUMBER = file_number
    MODEL_PATH = glb_path
    OUTPUT_STL = f"{FILE_NUMBER}.stl"

        # Base erzeugen
    create_baseplate(TARGET_RADIUS, BASE_THICKNESS, BASE_STL)

    # Text als STL erzeugen
    generate_text_stl(FILE_NUMBER, TEXT_HEIGHT, TEXT_SCAD, TEXT_STL)

    # Text-Mesh laden und skalieren
    textmesh = trimesh.load(TEXT_STL)
    textmesh.apply_translation(-textmesh.bounds[0])  # in ersten Quadrant verschieben

    text_width = textmesh.bounds[1, 0]
    scale = (2 * TARGET_RADIUS - TEXT_MARGIN) / text_width
    textmesh.apply_scale([scale, scale, 1])
    textmesh.apply_translation([
        -textmesh.bounds[1, 0] / 2,
        -textmesh.bounds[1, 1] / 2,
        -TEXT_HEIGHT / 2
    ])
    textmesh.export(TEXT_STL)

    # Gravierte Base erzeugen
    scad_boolean_difference(BASE_STL, TEXT_STL, ENGRAVED_STL)

    # Originalmodell laden und vorbereiten
    mesh = trimesh.load(MODEL_PATH)
    base = trimesh.load(ENGRAVED_STL)

    # Modell um X-Achse rotieren
    rot_x_90 = trimesh.transformations.rotation_matrix(
        np.deg2rad(90), direction=[1, 0, 0], point=[0, 0, 0]
    )
    mesh.apply_transform(rot_x_90)

    # Modell auf Basegröße skalieren
    bounds = mesh.bounds
    radius = np.max(np.abs(bounds[:2, :2]))
    mesh.apply_scale(TARGET_RADIUS / radius)

    # Modell über Base setzen
    z_min = mesh.bounds[0][2]
    mesh.apply_translation([0, 0, -z_min + BASE_THICKNESS - 0.6])

    # Kombinieren und exportieren
    combined = trimesh.util.concatenate([base, mesh])
    output_path = f"{output_folder}/{OUTPUT_STL}"
    combined.export(output_path)
    print(f"✅ Fertiges Modell exportiert: {output_path}")
    return output_path
