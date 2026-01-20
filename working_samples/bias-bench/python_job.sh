#!/bin/bash
#SBATCH --output=python_job.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# salloc --partition=debug --gres=gpu:L40:1 --mem=512G

# # python generate_stereoset_predictions.py state-spaces/transformerpp-2.7b
# # python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions.json --output_file results.json

# python generate_stereoset_predictions.py state-spaces/mamba-2.8b
# python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions.json --output_file results.json

# # python generate_stereoset_predictions.py state-spaces/mamba2-2.7b
# # python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions.json --output_file results.json

python generate_stereoset_predictions.py state-spaces/transformerpp-2.7b predictions_stereoset_transformerpp.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_transformerpp.json --output_file results_stereoset_transformerpp.json

python generate_stereoset_predictions.py state-spaces/mamba-2.8b predictions_stereoset_mamba.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_mamba.json --output_file results_stereoset_mamba.json

python generate_stereoset_predictions.py state-spaces/mamba2-2.7b predictions_stereoset_mamba2.json
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions_stereoset_mamba2.json --output_file results_stereoset_mamba2.json
