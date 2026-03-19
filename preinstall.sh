#!/bin/bash
# Force-remove the conflicting libraries
pip uninstall -y gradio huggingface_hub
# Install the specific compatible pair
pip install huggingface_hub==0.24.0 gradio==5.0.1