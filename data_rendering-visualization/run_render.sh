#!/bin/bash

# activate wsl environment 
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate WOMD

echo "Waymo Dataset Renderer"
echo "1. Generate Static Plot (PNG)"
echo "2. Generate Video Animation (MP4)"
read -p "Select an option: " choice

if [ "$choice" == "1" ]; then
	python render_scenario.py --mode static
elif [ "$choice" == "2" ]; then
	python render_scenario.py --mode video
else 
	echo "Not valid"
fi 
