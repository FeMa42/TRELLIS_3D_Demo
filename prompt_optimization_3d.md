# Prompt Optimization Guide for Text-to-3D Generation

## For TRELLIS 3D Demo Pipeline

### Table of Contents

1. [Overview](#overview)
2. [Available Prompt Enhancement Models](#available-prompt-enhancement-models)
3. [3D-Specific Prompt Optimization](#3d-specific-prompt-optimization)
4. [Implementation Guide](#implementation-guide)
5. [Best Practices from Research](#best-practices-from-research)
6. [Integration with TRELLIS](#integration-with-trellis)

---

## Overview

This guide covers prompt optimization techniques specifically for text-to-3D generation pipelines, where prompts are used to generate images (via FLUX/Qwen) that are then converted to 3D models using TRELLIS.

### Key Difference from Standard Image Generation

- **Goal**: Generate images optimized for 3D reconstruction, not artistic quality
- **Focus**: Geometric clarity, consistent lighting, and reconstructable views
- **Avoid**: Artistic effects, complex shadows, extreme perspectives

---

## Available Prompt Enhancement Models

### 1. Lightweight T5-Based Models (~250MB VRAM)

#### gokaygokay/Flux-Prompt-Enhance

**Best for**: General prompt enhancement with good 3D awareness

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "gokaygokay/Flux-Prompt-Enhance"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

enhancer = pipeline('text2text-generation', 
                   model=model, 
                   tokenizer=tokenizer, 
                   repetition_penalty=1.2, 
                   device=device)

# Usage
prefix = "enhance prompt: "
short_prompt = "a red sports car"
enhanced = enhancer(prefix + short_prompt, max_length=256)
print(enhanced[0]['generated_text'])
```

#### imranali291/flux-prompt-enhancer

**Best for**: Stable Diffusion/FLUX specific enhancement

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("imranali291/flux-prompt-enhancer")
model = T5ForConditionalGeneration.from_pretrained("imranali291/flux-prompt-enhancer")

input_text = "futuristic robot"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

output = model.generate(
    input_ids, 
    max_length=128, 
    do_sample=True, 
    top_p=0.9, 
    temperature=0.7,
    repetition_penalty=2.5
)
                       
enhanced_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
```

### 2. Instruction-Following Models (1-2GB VRAM)

#### Gemma-2-2B

**Pros**: Better instruction following than older versions
**Cons**: May add meta-commentary, needs post-processing

```python
from transformers import pipeline
import torch

pipe = pipeline("text-generation", 
                model="google/gemma-2-2b-it",
                torch_dtype=torch.bfloat16,
                device_map="auto")

messages = [{
    "role": "user",
    "content": f"Transform into 3D render prompt. Output ONLY the prompt: {user_input}"
}]

result = pipe(messages, max_new_tokens=100, temperature=0.7)
```

#### Qwen2.5-1.5B-Instruct  

**Best for**: Excellent instruction following, minimal meta-commentary

```python
from transformers import pipeline

pipe = pipeline("text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto")

messages = [
    {"role": "system", "content": "You are a prompt enhancer. Output only the enhanced prompt."},
    {"role": "user", "content": f"Enhance for 3D rendering: {prompt}"}
]

result = pipe(messages, max_new_tokens=100, temperature=0.7)
```

### 3. Legacy Model (Still Useful)

#### microsoft/Promptist (Archived)

- Based on GPT-2, ~500MB
- Trained specifically for Stable Diffusion 1.x
- Still works but outdated compared to newer options

---

## 3D-Specific Prompt Optimization

### Critical Elements for 3D-Ready Images

#### 1. Background

```python
backgrounds_3d = [
    "pure white background",      # Best for segmentation
    "neutral gray background",    # Good alternative
    "clean gradient background",   # Acceptable
    # AVOID: complex backgrounds, outdoor scenes
]
```

#### 2. Viewing Angles

```python
optimal_views = [
    "3/4 view angle",          # Best - shows depth + details
    "45 degree angle",         # Good for most objects
    "isometric view",          # Great for architectural/furniture
    "front facing view",       # Good for characters/faces
    # AVOID: extreme angles, bird's eye view, worm's eye view
]
```

#### 3. Lighting

```python
lighting_3d = [
    "soft studio lighting",    # Even illumination
    "ambient lighting",         # Reduces harsh shadows
    "diffused lighting",        # Good for materials
    "rim lighting",            # Helps with edge definition
    # AVOID: dramatic lighting, sunset/sunrise, harsh shadows
]
```

#### 4. Style Modifiers

```python
style_3d = [
    "3D render",
    "octane render",
    "unreal engine",
    "product visualization",
    "ZBrush sculpt",
    "CGI quality",
    # AVOID: "oil painting", "watercolor", "sketch"
]
```

### Prompt Structure Formula

```
[Object Description] + [View] + [Background] + [Lighting] + [Quality] + [Style]
```

Example:

```
"red sports car, 3/4 view angle, white background, studio lighting, highly detailed, 3D render"
```

---

## Implementation Guide

### Complete 3D Prompt Enhancement System

```python
import re
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TRELLIS3DPromptEnhancer:
    """
    Comprehensive prompt enhancement system for text-to-3D generation
    Optimized for TRELLIS pipeline with FLUX/Qwen image generation
    """
    
    def __init__(
        self, 
        use_llm_enhancement: bool = True,
        model_name: str = "gokaygokay/Flux-Prompt-Enhance",
        device: str = None
    ):
        self.use_llm_enhancement = use_llm_enhancement
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_llm_enhancement:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        
        # 3D-specific vocabulary
        self._init_3d_vocabulary()
    
    def _init_3d_vocabulary(self):
        """Initialize 3D-specific prompt components"""
        self.modifiers_3d = {
            "lighting": [
                "soft studio lighting",
                "even illumination", 
                "ambient lighting",
                "no harsh shadows"
            ],
            "background": [
                "pure white background",
                "neutral background",
                "clean background",
                "isolated object"
            ],
            "view": [
                "3/4 view angle",
                "three-quarter view",
                "45 degree angle",
                "isometric view"
            ],
            "quality": [
                "highly detailed",
                "sharp focus",
                "high resolution",
                "clear edges"
            ],
            "style": [
                "3D render",
                "octane render",
                "product visualization",
                "CGI quality"
            ]
        }
        
        # Object-specific templates
        self.templates = {
            "character": "{desc}, T-pose, front facing, {bg}, character model, game asset",
            "vehicle": "{desc}, 3/4 view, {bg}, {light}, detailed model, no motion blur",
            "furniture": "{desc}, isometric view, {bg}, product visualization, architectural",
            "animal": "{desc}, standing pose, side view, {bg}, 3D sculpt, detailed fur",
            "product": "{desc}, hero angle, {bg}, {light}, product photography, CAD render",
            "general": "{desc}, {view}, {bg}, {light}, {quality}, {style}"
        }
        
        # Words to avoid for 3D
        self.avoid_words = [
            "artistic", "abstract", "painterly", "sketchy", 
            "dramatic lighting", "sunset", "atmospheric",
            "motion blur", "depth of field", "bokeh"
        ]
    
    def enhance(
        self, 
        user_prompt: str, 
        object_type: str = "general",
        target_model: str = "flux",
        use_template: bool = True
    ) -> Dict[str, str]:
        """
        Main enhancement pipeline
        
        Args:
            user_prompt: Original user input
            object_type: Type of object (character, vehicle, etc.)
            target_model: Target image model (flux, qwen, etc.)
            use_template: Whether to use object templates
        
        Returns:
            Dictionary with enhanced prompts and metadata
        """
        
        # Stage 1: Clean and normalize input
        cleaned = self._clean_input(user_prompt)
        
        # Stage 2: LLM Enhancement (if enabled)
        if self.use_llm_enhancement:
            llm_enhanced = self._llm_enhance(cleaned)
        else:
            llm_enhanced = cleaned
        
        # Stage 3: Apply template (if applicable)
        if use_template and object_type in self.templates:
            templated = self._apply_template(llm_enhanced, object_type)
        else:
            templated = llm_enhanced
        
        # Stage 4: Add 3D modifiers
        with_modifiers = self._add_3d_modifiers(templated)
        
        # Stage 5: Remove problematic terms
        cleaned_3d = self._remove_problematic_terms(with_modifiers)
        
        # Stage 6: Format for target model
        final = self._format_for_model(cleaned_3d, target_model)
        
        # Stage 7: Add weighted attention (optional)
        weighted = self._create_weighted_version(final)
        
        return {
            "original": user_prompt,
            "enhanced": final,
            "weighted": weighted,
            "object_type": object_type,
            "target_model": target_model
        }
    
    def _clean_input(self, prompt: str) -> str:
        """Clean and normalize user input"""
        # Remove extra whitespace
        prompt = ' '.join(prompt.split())
        # Remove special characters that might cause issues
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        return prompt.strip()
    
    def _llm_enhance(self, prompt: str) -> str:
        """Use T5 model to enhance prompt"""
        input_text = f"enhance prompt: {prompt}"
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        enhanced = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up artifacts
        enhanced = re.sub(r'^.*?:', '', enhanced).strip()
        return enhanced
    
    def _apply_template(self, prompt: str, object_type: str) -> str:
        """Apply object-specific template"""
        template = self.templates[object_type]
        
        # Select appropriate modifiers
        bg = self.modifiers_3d["background"][0]
        light = self.modifiers_3d["lighting"][0]
        view = self.modifiers_3d["view"][0]
        quality = self.modifiers_3d["quality"][0]
        style = self.modifiers_3d["style"][0]
        
        return template.format(
            desc=prompt,
            bg=bg,
            light=light,
            view=view,
            quality=quality,
            style=style
        )
    
    def _add_3d_modifiers(self, prompt: str) -> str:
        """Add essential 3D modifiers if not present"""
        prompt_lower = prompt.lower()
        additions = []
        
        # Check for essential components
        if "background" not in prompt_lower:
            additions.append(self.modifiers_3d["background"][0])
        
        if not any(v.lower() in prompt_lower for v in self.modifiers_3d["view"]):
            additions.append(self.modifiers_3d["view"][0])
        
        if "light" not in prompt_lower:
            additions.append(self.modifiers_3d["lighting"][0])
        
        if not any(word in prompt_lower for word in ["detailed", "quality", "sharp"]):
            additions.append(self.modifiers_3d["quality"][0])
        
        if additions:
            return f"{prompt}, {', '.join(additions)}"
        return prompt
    
    def _remove_problematic_terms(self, prompt: str) -> str:
        """Remove terms that hurt 3D reconstruction"""
        for term in self.avoid_words:
            prompt = re.sub(rf'\b{term}\b', '', prompt, flags=re.IGNORECASE)
        
        # Clean up extra commas and spaces
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        return prompt.strip()
    
    def _format_for_model(self, prompt: str, model_type: str) -> str:
        """Add model-specific formatting"""
        model_suffixes = {
            "flux": ", octane render, unreal engine 5, photorealistic",
            "qwen": ", professional 3D visualization, clean composition",
            "gemini": ", high quality 3D render, product shot",
            "default": ", 3D render, professional quality"
        }
        
        suffix = model_suffixes.get(model_type, model_suffixes["default"])
        return f"{prompt}{suffix}"
    
    def _create_weighted_version(self, prompt: str) -> str:
        """Create version with attention weights for critical terms"""
        # Emphasize object and key 3D terms
        weighted = prompt
        
        # Add weights to critical terms
        replacements = [
            (r'\b(white background)\b', r'(\1:1.5)'),
            (r'\b(3/4 view|three-quarter view)\b', r'(\1:1.3)'),
            (r'\b(studio lighting)\b', r'(\1:1.2)'),
            (r'\b(highly detailed)\b', r'(\1:1.3)'),
        ]
        
        for pattern, replacement in replacements:
            weighted = re.sub(pattern, replacement, weighted, flags=re.IGNORECASE)
        
        return weighted

    def batch_enhance(self, prompts: List[str], **kwargs) -> List[Dict[str, str]]:
        """Enhance multiple prompts efficiently"""
        return [self.enhance(p, **kwargs) for p in prompts]

# Utility functions for prompt post-processing

def clean_gemma_output(text: str) -> str:
    """Remove common meta-commentary from Gemma outputs"""
    prefixes_to_remove = [
        r"^Here'?s? (?:the |an? )?(?:enhanced |improved )?prompt:?\s*",
        r"^This (?:is |would be |creates ):?\s*",
        r"^(?:The )?(?:Enhanced |Improved )?prompt:?\s*",
    ]
    
    for pattern in prefixes_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Take first sentence if it's a valid prompt
    sentences = text.split('.')
    if sentences[0].strip() and len(sentences[0]) > 20:
        return sentences[0].strip()
    
    return text.strip()

def validate_3d_prompt(prompt: str) -> Dict[str, bool]:
    """Validate if prompt has necessary 3D elements"""
    checks = {
        "has_background": any(word in prompt.lower() for word in ["background", "backdrop"]),
        "has_view": any(word in prompt.lower() for word in ["view", "angle", "facing", "pose"]),
        "has_lighting": any(word in prompt.lower() for word in ["light", "lit", "illumin"]),
        "has_quality": any(word in prompt.lower() for word in ["detailed", "quality", "resolution"]),
        "no_problematic": not any(word in prompt.lower() for word in ["abstract", "artistic", "painterly"])
    }
    checks["is_valid"] = all(checks.values())
    return checks
```

---

## Best Practices from Research

### 1. Dynamic Prompt Optimization (PAE - CVPR 2024)

- **Finding**: Adjusting word weights and injection timesteps improves quality
- **Application**: Use attention weighting for critical 3D terms
- **Implementation**: `(white background:1.5)` emphasizes importance

### 2. Reinforcement Learning Approach

- **Finding**: RL-based optimization improves aesthetic and relevance scores
- **Application**: Iteratively refine prompts based on 3D reconstruction success
- **Recommendation**: Track which prompts produce best 3D models

### 3. Prompt Search Optimization (DPO-DIFF - ICML 2024)

- **Finding**: Search-based methods can optimize both positive and negative prompts
- **For 3D**: Avoid negative prompts - they create inconsistencies harmful to reconstruction

### 4. Key Parameters for 3D Generation

```python
# Optimal generation parameters for 3D-ready images
generation_params = {
    "temperature": 0.3-0.5,    # Lower = more consistent
    "top_p": 0.8-0.9,          # Focused sampling
    "cfg_scale": 7-9,          # Strong prompt adherence
    "num_inference_steps": 30-50,  # Balance quality/speed
    "negative_prompt": None     # Avoid for 3D consistency
}
```

### 5. Multi-View Strategy

Generate multiple angles and select best for 3D:

```python
def generate_multiview(prompt_base, enhancer, image_model):
    views = ["front view", "3/4 view", "side view", "back view"]
    images = []
    
    for view in views:
        enhanced = enhancer.enhance(f"{prompt_base}, {view}")
        image = generate_image(enhanced["enhanced"], model=image_model)
        images.append(image)
    
    # Select best based on clarity, contrast, object isolation
    return select_best_for_3d(images)
```

---

## Integration with TRELLIS

### Basic Integration

```python
# app_text.py modification

from prompt_enhancer_3d import TRELLIS3DPromptEnhancer

class Text23DPipeline:
    def __init__(self, config):
        # Initialize prompt enhancer
        self.prompt_enhancer = TRELLIS3DPromptEnhancer(
            use_llm_enhancement=True,
            model_name="gokaygokay/Flux-Prompt-Enhance"
        )
        
        # Existing TRELLIS initialization
        self.trellis = self._init_trellis(config)
        self.image_generator = self._init_image_model(config)
    
    def generate_3d(self, text_prompt, options=None):
        """Complete text-to-3D pipeline"""
        
        # Step 1: Enhance prompt
        enhanced_result = self.prompt_enhancer.enhance(
            text_prompt,
            object_type=options.get("object_type", "general"),
            target_model=options.get("image_model", "flux")
        )
        
        print(f"Original: {enhanced_result['original']}")
        print(f"Enhanced: {enhanced_result['enhanced']}")
        
        # Step 2: Generate image
        image = self.generate_image(
            enhanced_result['enhanced'],
            model=options.get("image_model", "flux")
        )
        
        # Step 3: Generate 3D with TRELLIS
        mesh = self.trellis.generate(image)
        
        return {
            "mesh": mesh,
            "image": image,
            "prompts": enhanced_result
        }
```

### Streamlit Integration

```python
# streamlit_app.py

import streamlit as st
from prompt_enhancer_3d import TRELLIS3DPromptEnhancer

@st.cache_resource
def load_enhancer():
    return TRELLIS3DPromptEnhancer(use_llm_enhancement=True)

def main():
    st.title("TRELLIS 3D Generation with Prompt Enhancement")
    
    enhancer = load_enhancer()
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        user_prompt = st.text_area("Enter object description:", 
                                   placeholder="e.g., a red sports car")
        
        object_type = st.selectbox("Object Type:", 
                                   ["general", "character", "vehicle", 
                                    "furniture", "animal", "product"])
        
        image_model = st.selectbox("Image Model:", 
                                   ["flux", "qwen", "gemini"])
    
    with col2:
        st.subheader("Enhancement Options")
        use_llm = st.checkbox("Use LLM Enhancement", value=True)
        use_template = st.checkbox("Use Object Template", value=True)
        show_weighted = st.checkbox("Show Weighted Version", value=False)
    
    if st.button("Generate 3D"):
        # Enhance prompt
        enhanced = enhancer.enhance(
            user_prompt,
            object_type=object_type,
            target_model=image_model,
            use_template=use_template
        )
        
        # Display prompts
        st.subheader("Prompt Enhancement")
        st.text(f"Original: {enhanced['original']}")
        st.text(f"Enhanced: {enhanced['enhanced']}")
        if show_weighted:
            st.text(f"Weighted: {enhanced['weighted']}")
        
        # Generate 3D (existing pipeline)
        with st.spinner("Generating 3D model..."):
            result = generate_3d_model(enhanced['enhanced'])
            display_3d_result(result)

if __name__ == "__main__":
    main()
```

### Environment Variables Setup

```bash
# app_text.sh modification

#!/bin/bash

# Prompt enhancement settings
export USE_PROMPT_ENHANCEMENT=true
export PROMPT_MODEL="gokaygokay/Flux-Prompt-Enhance"
export PROMPT_CACHE_DIR="./cache/prompts"

# Existing TRELLIS settings
export ENABLE_TRELLIS_CPU_OFFLOAD=${ENABLE_TRELLIS_CPU_OFFLOAD:-true}
export ENABLE_IMAGE_CPU_OFFLOAD=${ENABLE_IMAGE_CPU_OFFLOAD:-true}

# Run app
python app_text.py
```

---

## Usage Examples

### Example 1: Simple Object

```python
enhancer = TRELLIS3DPromptEnhancer()

result = enhancer.enhance("coffee mug")
print(result["enhanced"])
# Output: "coffee mug, 3/4 view angle, pure white background, soft studio lighting, highly detailed, octane render, unreal engine 5, photorealistic"
```

### Example 2: Complex Character

```python
result = enhancer.enhance(
    "wizard with long beard holding a staff",
    object_type="character"
)
print(result["enhanced"])
# Output: "wizard with long beard holding a staff, T-pose, front facing, pure white background, character model, game asset, soft studio lighting, highly detailed, octane render, unreal engine 5, photorealistic"
```

### Example 3: Batch Processing

```python
prompts = [
    "red sports car",
    "wooden chair",
    "golden retriever dog"
]

results = enhancer.batch_enhance(prompts, target_model="flux")
for r in results:
    print(f"{r['original']} -> {r['enhanced'][:50]}...")
```

### Example 4: Validation Check

```python
prompt = "abstract artistic painting of a sunset"
validation = validate_3d_prompt(prompt)
print(validation)
# Output: {'has_background': False, 'has_view': False, 'has_lighting': True, 'has_quality': False, 'no_problematic': False, 'is_valid': False}

if not validation["is_valid"]:
    # Enhance it for 3D
    enhanced = enhancer.enhance(prompt)
    print(f"Fixed: {enhanced['enhanced']}")
```

---

## Performance Optimization

### Memory Management

```python
# For limited VRAM setups
class LowMemoryEnhancer(TRELLIS3DPromptEnhancer):
    def __init__(self):
        super().__init__(use_llm_enhancement=True)
        # Use CPU offloading
        self.model = self.model.half()  # FP16
        
    def enhance_with_offload(self, prompt):
        # Move to GPU only when needed
        self.model.to("cuda")
        result = self.enhance(prompt)
        self.model.to("cpu")  # Offload after use
        torch.cuda.empty_cache()
        return result
```

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_enhance(prompt: str, object_type: str = "general") -> str:
    """Cache enhanced prompts to avoid redundant processing"""
    enhancer = get_global_enhancer()  # Singleton instance
    return enhancer.enhance(prompt, object_type)["enhanced"]
```

---

## Troubleshooting

### Common Issues and Solutions

1. **Meta-commentary in output**
   - Solution: Use `clean_gemma_output()` function
   - Alternative: Switch to Qwen model

2. **Poor 3D reconstruction**
   - Check: Run `validate_3d_prompt()` on enhanced prompt
   - Fix: Ensure white background and proper viewing angle

3. **Out of memory**
   - Use T5 models instead of larger instruction models
   - Enable CPU offloading
   - Reduce batch size

4. **Inconsistent results**
   - Lower temperature (0.3-0.5)
   - Increase repetition penalty
   - Use templates for consistency

---

## Model Recommendations Summary

| Model | VRAM | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| gokaygokay/Flux-Prompt-Enhance | ~250MB | Fast | High | General 3D enhancement |
| imranali291/flux-prompt-enhancer | ~250MB | Fast | High | FLUX-specific |
| Qwen2.5-1.5B-Instruct | ~1.5GB | Medium | High | Instruction following |
| Gemma-2-2B | ~2GB | Medium | Medium | Fallback option |

---

## References

1. **PAE (CVPR 2024)**: Dynamic Prompt Optimizing for Text-to-Image Generation
2. **DPO-DIFF (ICML 2024)**: Discrete Prompt Optimization for Diffusion Models  
3. **Promptist**: Original prompt optimization model by Microsoft
4. **TRELLIS**: Microsoft's Image-to-3D generation framework

---

## Quick Start Code

```python
# Complete minimal example
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("gokaygokay/Flux-Prompt-Enhance")
model = AutoModelForSeq2SeqLM.from_pretrained("gokaygokay/Flux-Prompt-Enhance")

# Enhance prompt
def quick_enhance_3d(prompt):
    input_text = f"enhance prompt: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, temperature=0.7)
    enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add 3D essentials
    if "background" not in enhanced.lower():
        enhanced += ", white background"
    if "view" not in enhanced.lower():
        enhanced += ", 3/4 view angle"
    
    return enhanced + ", 3D render, studio lighting"

# Use it
result = quick_enhance_3d("red dragon")
print(result)
```
