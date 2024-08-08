# make sure you have COLMAP and Nerfstudio installed
# set working directory in config.yaml
# put convert.py in the folder before the working directory
# then run the following commands:
python3 extract_frames.py
ns-process-data images --data input/ --output-dir ./ --camera-type perspective --skip-image-processing --no-gpu --verbose
rm sparse_pc.ply transforms.json
mv colmap/ distorted/
python3 ../convert.py --source_path . --skip_matching --resize
ns-process-data images --data images/ --output-dir . --skip-image-processing --skip-colmap --colmap-model-path sparse/0/
