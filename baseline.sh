#/bin/bash
all_tasks="cifar10"

for task in $all_tasks
do    
    # 3 runs sanity check
    bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
    bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
    bash multi_run_experiment.sh final_exp_logs/sanity_check/$task $task 1 0 --agent-type Agent
done

