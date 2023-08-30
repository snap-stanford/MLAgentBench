#/bin/bash

source ~/.bashrc
conda activate llm

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

extra_args="${@}"
ts=$(date +%s)
folder=interactive_logs/$ts
python=$(which python)

# echo "extra_args: $extra_args"
echo "Logs will be saved to $folder"

mkdir -p $folder

python -u runner.py --interactive --python $python --max_steps 100 --device 0 --log_dir $folder $extra_args 2> $folder/log 
