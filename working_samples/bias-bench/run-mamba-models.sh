#!/bin/bash

python generate_stereoset_predictions.py state-spaces/transformerpp-2.7b predictions_stereoset_transformerpp.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_transformerpp.json --output_file results_stereoset_transformerpp.json

python generate_stereoset_predictions.py state-spaces/mamba-2.8b predictions_stereoset_mamba.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_mamba.json --output_file results_stereoset_mamba.json

python generate_stereoset_predictions.py state-spaces/mamba2-2.7b predictions_stereoset_mamba2.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_mamba2.json --output_file results_stereoset_mamba2.json
