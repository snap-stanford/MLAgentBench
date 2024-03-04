#/bin/bash

all_tasks="cifar10_fix"
log_dir=final_exp_logs
models="no_retrieval_gpt-4-0125-preview"

for model in $models
do
    for task in $all_tasks
    do
        echo "python -m MLAgentBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json"
        python -m MLAgentBench.eval --log-folder $log_dir/$model/$task --task $task --output-file ${model}_${task}.json --eval-intermediate 
        # add --eval-intermediate to evaluate intermediate steps 
    done
done