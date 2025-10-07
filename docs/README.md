# Documentation

This directory contains technical documentation and blog posts about the CVMT project.

## Contents

### Blog Post: Machine Learning Strategy

**File**: `machine-learning-strategy.md`

A comprehensive, blog-ready document explaining the machine learning architecture, computer vision techniques, and training strategies used in this project. This document is designed to showcase your technical expertise and can be published directly on your website or blog.

**Topics Covered:**
- Multi-task U-Net architecture with transfer learning
- Heatmap-based landmark detection
- Data augmentation and preprocessing strategies
- Training methodology with PyTorch Lightning
- Evaluation metrics and model verification
- End-to-end inference pipeline
- Geometric analysis for bone age classification

### Generating Figures

**File**: `generate_blog_figures.py`

A Python script that generates all the visualizations referenced in the blog post using the actual cvmt codebase.

**Usage:**

```bash
# From the root directory of the repository
python docs/generate_blog_figures.py
```

**Generated Figures:**
1. `unet_architecture.png` - Model architecture diagram
2. `heatmap_visualization.png` - Gaussian heatmap representations
3. `augmentation_examples.png` - Data augmentation transformations
4. `model_performance.png` - MRE distribution and prediction examples
5. `inference_result.png` - End-to-end inference visualization


All figures are saved to `docs/images/` directory.

**Requirements:**
- All dependencies from `pyproject.toml` installed
- Optional: `torchview` for architecture visualization (`pip install torchview`)

## Publishing the Blog Post

The blog post (`machine-learning-strategy.md`) is ready for publication with:
- ✅ Engaging introduction explaining real-world impact
- ✅ Technical depth with code examples
- ✅ Visual elements (figures with captions)
- ✅ Clear explanations using simple English
- ✅ Comprehensive conclusion with lessons learned
- ✅ Call-to-action and acknowledgments

### Steps to Publish:

1. **Generate figures** (optional, if you want real visualizations):
   ```bash
   python docs/generate_blog_figures.py
   ```

2. **Convert to HTML** (if needed):
   ```bash
   # Using pandoc
   pandoc machine-learning-strategy.md -o machine-learning-strategy.html --standalone --toc

   # Or using a static site generator like Jekyll, Hugo, etc.
   ```

3. **Copy to your website**:
   - If using a static site generator, copy the markdown file to your posts directory
   - If using a CMS, copy the converted HTML
   - Upload images to your website's image directory

4. **Update image paths**: Change `docs/images/` paths to your website's image URL structure


## Image Directory

The `images/` subdirectory contains:
- Generated figures from the script
- The McNamara & Franchi six stages diagram (from the main README)
- Any additional illustrations or screenshots

## License

This documentation is part of the CVMT project and follows the same MIT license as the main repository.

## Questions or Feedback?

For questions about the documentation or suggestions for improvement:
- Open an issue on GitHub
- Submit a pull request with improvements
- Contact via the repository's contact methods

---

**Last Updated**: October 2025
