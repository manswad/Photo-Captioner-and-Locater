# Image Search System

## Overview

A two-part system for intelligent image management and search using natural language descriptions. The system consists of two independent components that can run on different machines:

1. **Describer** (`describer.py`): A heavyweight component that generates natural language descriptions for images using the BLIP model
2. **Searcher** (`searcher.py`): A lightweight GUI application for searching images using text descriptions

## System Architecture

The system is designed to be modular, allowing for distributed deployment:

* The Describer component can run on a powerful machine (e.g., a server with GPU) to process images and generate descriptions
* The Searcher component can run on lightweight devices (e.g., Raspberry Pi, old laptops) to search through the processed images

## Components

### Describer (`describer.py`)

Processes a directory of images to:

* Generate natural language descriptions using the BLIP model
* Create a standardized naming scheme for images
* Save processed images to a new directory
* Generate an index CSV file containing filenames and descriptions

#### Dependencies
```
pandas
Pillow
transformers
torch
torchvision
```

### Searcher (`searcher.py`)

Provides a GUI interface to:

* Search through processed images using natural language queries
* Display top matching results with similarity scores
* Preview and select images

#### Dependencies
```
pandas
Pillow
scikit-learn
tkinter (usually comes with Python)
```

## Installation

### Full System (Both Components)

For GPUs install torch from [here](https://pytorch.org/get-started/locally/).

```bash
pip install pandas Pillow transformers torch torchvision scikit-learn
```

### Lightweight Search-Only System

```bash
pip install pandas Pillow scikit-learn
```

## Usage

1. First, process your images using the Describer:

   ```bash
   python describer.py
   ```

   This will:
   * Read images from the `photos` directory
   * Process them and save to `described` directory
   * Create an `index.csv` file with descriptions
   * Some samples have already been added, you can delete them if you want

2. Use the Searcher to find images:

   ```bash
   python searcher.py
   ```

   This will launch a GUI where you can:
   * Enter natural language descriptions
   * View matching images with similarity scores
   * Select images for use

## Distributed Setup

You can run this system in a distributed manner:

1. Run `describer.py` on a powerful machine:
   * Process all images
   * Generate the `described` directory and `index.csv`

2. Copy the `described` directory (with images and CSV) to lightweight devices

3. Run `searcher.py` on lightweight devices:
   * Only requires the processed images and CSV
   * No heavy ML models needed
   * Works well on devices like Raspberry Pi

## Directory Structure

```
.
├── photos/          # Original images
├── described/       # Processed images
│   └── index.csv   # Image descriptions and filenames
├── describer.py    # Image processing script
└── searcher.py     # Search GUI application
```

## Features

* Natural language image description generation
* Text-based image search using TF-IDF and cosine similarity
* Modern GUI interface with image previews
* Modular design allowing for distributed deployment
* Lightweight search component suitable for resource-constrained devices

## Notes

* The Describer component requires significant computational resources due to the BLIP model
* The Searcher component is very lightweight and can run on minimal hardware
* Images are automatically renamed and organized during processing
* The system supports JPG, PNG, and JPEG formats

## Future Improvements

* Add batch processing capabilities to Describer
* Implement caching for faster search results
* Add support for more image formats
* Include image metadata in the search index
* Add export functionality for search results 