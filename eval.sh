#/bin/bash

all_tasks="cifar10 imdb"
log_dir=final_exp_logs
models="claude2.1 gpt-4-0125-preview gemini-pro"

for model in $models
do
    for task in $all_tasks
    do
        echo "python -m MLAgentBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json"
        python -m MLAgentBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json 
        # add --eval-intermediate to evaluate intermediate steps 
    done
done