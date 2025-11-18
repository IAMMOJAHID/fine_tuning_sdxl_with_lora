## 🎯 Fine-Tuning Overview

**Objective**: Fine-tune Stable Diffusion XL on the Naruto dataset to learn the distinctive anime art style, characters, and aesthetic while operating within 16GB VRAM constraints.

**Strategy**: Use LoRA (Low-Rank Adaptation) to efficiently adapt the base model to the Naruto domain without full retraining, employing multiple memory optimization techniques to handle SDXL's large size.


## 💾 Applied VRAM Optimization Techniques

### 1. **LoRA (Low-Rank Adaptation)**
LoRA freezes the original model weights and injects trainable rank-decomposition matrices into attention layers. Instead of updating all parameters, LoRA updates only small adapter matrices. Full fine-tuning would require ~25+ GB VRAM but after LoRA, only ~4-5% of parameters are trainabl. Thus, approximately 50-60% VRAM reduction.


2. **Mixed Precision Training (FP16)**

Uses 16-bit floating point instead of 32-bit for most operations. Faster computations on modern GPUs. Minimal precision loss, well-tolerated in diffusion models

### 3. **8-bit Adam Optimizer**

Compresses optimizer states from 32-bit to 8-bit using quantization. 4x memory reduction


### 4. **Gradient Checkpointing**
Trade computation for memory by recomputing activations during backward pass instead of storing them. A significant saving in activation memory.

### 5. **Memory-Efficient Attention (xformers)**
Optimized attention implementation that reduces memory footprint of self-attention layers using xformers. This increase the compustion speed and helpful in high resolution image (1024x1024).


## 🎪 Motivation Behind Technique Choices

### **Why LoRA over Full Fine-tuning?**
Train 10-50M parameters vs 5.2B parameters. Preserves base model capabilities. Small checkpoint files (~10-100MB vs ~6-12GB)

### **Why Resolution 1024?**

SDXL is trained natively at 1024x1024. Lower resolutions would:
- Waste model capacity
- Produce blurry results
- Require the model to "learn downscaling"

### **Why Batch Size 3?**
Large enough for stable gradients. Fits within 16GB with other optimizations. Good trade-off between speed and quality

### **Why Train Text Encoder?**
Make it learn Naruto-specific terms and concepts. Thus, Better understanding of anime-style descriptions. Text encoder is much smaller than U-Net and thus it consumes very less memory.


### **Why DreamBooth is Not Suitable:**
1.	DreamBooth is for subject-driven generation - it's designed to teach a model about a specific subject (like a person, pet, or object) using just 3-5 images of that subject.
2.	Given dataset has diverse content - the Naruto dataset contains many different characters, scenes, and concepts, each with their own captions.
3.	DreamBooth uses class images - it requires generating "class images" to preserve general knowledge, which doesn't align with your multi-concept dataset.
