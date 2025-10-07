# Gemini 2.5 Flash Image Generation Guide

A practical guide for implementing image generation with Gemini 2.5 Flash (aka "nano banana").

## Installation

```bash
pip install google-genai pillow
```

## Quick Start (Minimal)

The simplest possible implementation:

```python
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=["A photorealistic sunset over mountains"]
)

# Save the generated image
for part in response.parts:
    if part.inline_data:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("output.png")
```

---

## Implementation Patterns by Use Case

### 1. Research & Prototyping (Fast Iteration)

**When to use:** Quick experiments, testing prompts, single image generation

**Characteristics:**

- Non-streaming (simpler code)
- Minimal error handling
- PIL for easy image manipulation
- Direct, readable code

```python
from google import genai
from PIL import Image
from io import BytesIO

def generate_image(prompt: str, output_path: str = "output.png"):
    """Simple image generation for prototyping."""
    client = genai.Client(api_key="YOUR_API_KEY")
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt]
    )
    
    for part in response.parts:
        if part.inline_data:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(output_path)
            return image
    
    return None

# Usage
img = generate_image("A futuristic cityscape at night")
img.show()  # Quick preview
```

### 2. Production Applications (Robust)

**When to use:** Web services, APIs, batch processing, user-facing applications

**Characteristics:**

- Streaming for better responsiveness
- Comprehensive error handling
- Handles multiple images
- Proper logging

```python
from google import genai
from google.genai import types
import mimetypes
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_image_production(
    prompt: str,
    output_dir: str = "outputs",
    base_filename: str = "generated"
) -> list[str]:
    """Production-ready image generation with error handling."""
    
    client = genai.Client(api_key="YOUR_API_KEY")
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"]
    )
    
    Path(output_dir).mkdir(exist_ok=True)
    saved_files = []
    file_index = 0
    
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=config
        ):
            # Defensive null checks
            if (chunk.candidates is None or 
                chunk.candidates[0].content is None or 
                chunk.candidates[0].content.parts is None):
                continue
            
            part = chunk.candidates[0].content.parts[0]
            
            # Handle image data
            if part.inline_data and part.inline_data.data:
                inline_data = part.inline_data
                extension = mimetypes.guess_extension(inline_data.mime_type) or ".png"
                filename = f"{base_filename}_{file_index}{extension}"
                filepath = Path(output_dir) / filename
                
                with open(filepath, "wb") as f:
                    f.write(inline_data.data)
                
                saved_files.append(str(filepath))
                logger.info(f"Saved image: {filepath}")
                file_index += 1
            
            # Handle text responses (captions, etc.)
            elif hasattr(chunk, 'text') and chunk.text:
                logger.info(f"Caption: {chunk.text}")
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise
    
    return saved_files

# Usage
files = generate_image_production(
    "A serene Japanese garden in autumn",
    output_dir="my_images"
)
print(f"Generated {len(files)} images: {files}")
```

### 3. Batch Processing (Multiple Images)

**When to use:** Dataset generation, A/B testing prompts, creating image variations

```python
from google import genai
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

def generate_single(prompt: str, index: int) -> Tuple[int, Image.Image]:
    """Generate a single image (for parallel execution)."""
    client = genai.Client(api_key="YOUR_API_KEY")
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt]
    )
    
    for part in response.parts:
        if part.inline_data:
            return (index, Image.open(BytesIO(part.inline_data.data)))
    
    return (index, None)

def batch_generate(prompts: List[str], max_workers: int = 4) -> List[Image.Image]:
    """Generate multiple images in parallel."""
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_single, prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        results = [f.result() for f in futures]
    
    # Sort by original index
    results.sort(key=lambda x: x[0])
    return [img for _, img in results if img is not None]

# Usage
prompts = [
    "A red sports car",
    "A blue sports car",
    "A green sports car"
]

images = batch_generate(prompts)

for i, img in enumerate(images):
    img.save(f"car_{i}.png")
```

### 4. Interactive Applications (Real-time Feedback)

**When to use:** User interfaces, progressive loading, live preview

```python
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

def generate_with_progress(prompt: str, callback=None):
    """Generate image with progress callbacks for UI updates."""
    
    client = genai.Client(api_key="YOUR_API_KEY")
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"]
    )
    
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=config
    ):
        if (chunk.candidates is None or 
            chunk.candidates[0].content is None or 
            chunk.candidates[0].content.parts is None):
            continue
        
        part = chunk.candidates[0].content.parts[0]
        
        # Text updates (e.g., "Generating...", "Adding details...")
        if hasattr(chunk, 'text') and chunk.text:
            if callback:
                callback("text", chunk.text)
        
        # Final image
        if part.inline_data and part.inline_data.data:
            image = Image.open(BytesIO(part.inline_data.data))
            if callback:
                callback("image", image)
            return image
    
    return None

# Usage with Gradio/Streamlit-style callback
def ui_callback(data_type, data):
    if data_type == "text":
        print(f"Status: {data}")
    elif data_type == "image":
        print("Image ready!")
        data.show()

generate_with_progress(
    "A cyberpunk street scene",
    callback=ui_callback
)
```

---

## Configuration Options

### Aspect Ratios

Gemini 2.5 Flash Image supports 10 aspect ratios:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

# Available aspect ratios
aspect_ratios = [
    "1:1",    # Square
    "3:4",    # Portrait
    "4:3",    # Landscape
    "9:16",   # Vertical (social media stories)
    "16:9",   # Horizontal (cinematic)
    # And more...
]

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=["A mountain landscape"],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9"  # Cinematic format
        )
    )
)
```

### Response Modalities

Control whether you want text, images, or both:

```python
# Image only (no text caption)
config = types.GenerateContentConfig(
    response_modalities=["IMAGE"]
)

# Text and image
config = types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"]
)

# Text only (e.g., for image analysis without generation)
config = types.GenerateContentConfig(
    response_modalities=["TEXT"]
)
```

---

## Image Editing & Manipulation

### Basic Image Editing

```python
from google import genai
from PIL import Image

client = genai.Client(api_key="YOUR_API_KEY")

# Load input image
input_image = Image.open("original.jpg")

# Edit with natural language
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[
        "Make the background snowy and mountainous",
        input_image
    ]
)

# Save edited image
for part in response.parts:
    if part.inline_data:
        edited = Image.open(BytesIO(part.inline_data.data))
        edited.save("edited.png")
```

### Multi-Image Composition

Merge multiple images:

```python
img1 = Image.open("character.jpg")
img2 = Image.open("background.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[
        "Place the character from the first image into the scene from the second image",
        img1,
        img2
    ]
)
```

### Iterative Refinement

Keep a conversation context for iterative edits:

```python
from google.genai import types

# Start with base generation
messages = [
    types.Content(
        role="user",
        parts=[types.Part.from_text("A modern living room")]
    )
]

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=messages
)

# Get the generated image
generated_image = None
for part in response.parts:
    if part.inline_data:
        generated_image = Image.open(BytesIO(part.inline_data.data))

# Continue the conversation for refinement
messages.append(
    types.Content(
        role="model",
        parts=[types.Part.from_bytes(
            data=part.inline_data.data,
            mime_type=part.inline_data.mime_type
        )]
    )
)

messages.append(
    types.Content(
        role="user",
        parts=[types.Part.from_text("Add a large potted plant in the corner")]
    )
)

# Generate refined version
refined_response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=messages
)
```

---

## Prompt Engineering Tips

Based on the documentation, here are strategies for better results:

### 1. Describe Scenes, Not Keywords

```python
# ❌ Less effective
"cat, space suit, moon, buggy"

# ✅ More effective
"A photorealistic image of an orange tabby cat wearing a white NASA spacesuit, confidently driving a lunar rover across the moon's grey, cratered surface under a black starry sky"
```

### 2. Use Photography Terms for Realism

```python
prompt = """
A photorealistic close-up portrait of an elderly woman with weathered hands, 
crafting pottery on a spinning wheel. Shot with a 50mm lens at f/1.8, 
creating a shallow depth of field that blurs the background workshop. 
Soft natural light streams in from a window to the left, creating gentle shadows. 
The clay is wet and glistening. High detail on the texture of her hands and the clay.
"""
```

### 3. Style-Specific Templates

```python
# For illustrations
prompt = "A kawaii-style sticker of a happy red panda wearing a tiny bamboo hat. The design features bold, clean outlines, simple cel-shading, and a vibrant color palette. The background must be white."

# For technical/educational
prompt = "Create a clear, educational diagram showing the water cycle. Include labeled arrows showing evaporation, condensation, precipitation, and collection. Use a clean, minimalist style with a light blue color scheme."

# For artistic
prompt = "An impressionist painting in the style of Monet, depicting a garden in spring with blooming cherry blossoms, dappled sunlight, and loose, expressive brushstrokes in pastel colors."
```

---

## Error Handling Best Practices

```python
from google import genai
from google.genai.errors import APIError
import time

def generate_with_retry(
    prompt: str, 
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Image.Image:
    """Generate image with exponential backoff retry."""
    
    client = genai.Client(api_key="YOUR_API_KEY")
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt]
            )
            
            for part in response.parts:
                if part.inline_data:
                    return Image.open(BytesIO(part.inline_data.data))
            
            raise ValueError("No image generated in response")
        
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = backoff_factor ** attempt
            print(f"API error, retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
    return None
```

---

## Environment Variables Best Practice

```python
import os
from google import genai

# Set via environment variable
# export GEMINI_API_KEY="your-api-key-here"

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Or use python-dotenv for local development
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
```

---

## Cost Considerations

**Pricing:** $0.039 per image (1290 output tokens @ $30/1M tokens)

For cost-effective generation:

```python
def generate_cost_aware(prompts: List[str], budget: float = 1.0):
    """Generate images within a budget."""
    
    cost_per_image = 0.039
    max_images = int(budget / cost_per_image)
    
    if len(prompts) > max_images:
        print(f"Budget allows {max_images} images, truncating prompts")
        prompts = prompts[:max_images]
    
    images = []
    total_cost = 0.0
    
    for prompt in prompts:
        img = generate_image(prompt)
        if img:
            images.append(img)
            total_cost += cost_per_image
            print(f"Generated image. Total cost: ${total_cost:.3f}")
    
    print(f"Final cost: ${total_cost:.2f} for {len(images)} images")
    return images
```

---

## Comparison: When to Use Which Pattern

| Use Case | Pattern | Key Feature |
|----------|---------|-------------|
| Quick experiments | Quick Start | Minimal code, fast iteration |
| Research prototypes | Prototyping | PIL integration, easy manipulation |
| Production APIs | Production | Error handling, streaming, logging |
| Dataset generation | Batch Processing | Parallel execution |
| Web applications | Interactive | Progress callbacks, real-time updates |
| Image editing tools | Image Editing | Multi-image input, iterative refinement |

---

## Full Working Example: Simple CLI Tool

```python
#!/usr/bin/env python3
"""
Simple CLI tool for Gemini 2.5 Flash Image generation.
Usage: python generate.py "A sunset over mountains" --output sunset.png
"""

import argparse
import os
from google import genai
from PIL import Image
from io import BytesIO

def main():
    parser = argparse.ArgumentParser(description="Generate images with Gemini 2.5 Flash")
    parser.add_argument("prompt", help="Text description of image to generate")
    parser.add_argument("--output", "-o", default="output.png", help="Output filename")
    parser.add_argument("--aspect", "-a", default="1:1", help="Aspect ratio (e.g., 16:9)")
    parser.add_argument("--show", "-s", action="store_true", help="Display image after generation")
    
    args = parser.parse_args()
    
    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    print(f"Generating: {args.prompt}")
    print(f"Aspect ratio: {args.aspect}")
    
    # Generate
    from google.genai import types
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[args.prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=args.aspect)
        )
    )
    
    # Save
    for part in response.parts:
        if part.inline_data:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(args.output)
            print(f"✓ Saved to: {args.output}")
            
            if args.show:
                image.show()
            
            return
    
    print("✗ No image generated")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

- **Official Documentation:** <https://ai.google.dev/gemini-api/docs/image-generation>
- **Google AI Studio:** <https://aistudio.google.com> (test prompts in browser)
- **Model Pricing:** $0.039 per image
- **Rate Limits:** Check your API dashboard
- **Model Name:** `gemini-2.5-flash-image`

---

## Common Issues & Solutions

**Issue:** `No image in response`

```python
# Check response_modalities includes "IMAGE"
config = types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"]  # Not just ["TEXT"]
)
```

**Issue:** `API key not found`

```python
# Verify environment variable is set
import os
print(os.getenv("GEMINI_API_KEY"))  # Should not be None
```

**Issue:** `Low quality images`

```python
# Use detailed, descriptive prompts with photography terms
# Be specific about lighting, composition, style, details
```

**Issue:** `Slow generation in production`

```python
# Use streaming for better responsiveness
# Implement caching for repeated prompts
# Consider batch processing for multiple images
```

---

## Summary

Choose your implementation based on your needs:

- **Prototyping?** Use the Quick Start pattern
- **Building an API?** Use the Production pattern with streaming
- **Processing many images?** Use Batch Processing
- **Building a UI?** Use Interactive pattern with callbacks
- **Editing images?** Use Image Editing with multi-modal input

The key is matching complexity to requirements - start simple, add robustness when needed.
