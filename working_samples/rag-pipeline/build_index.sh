#!/bin/bash

cd /home/$(whoami)/anlp-hw2
python offline/03_index.py --hf_model_path hkunlp/instructor-xl --instruct_embedding --process_table