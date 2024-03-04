#/bin/bash

# example tasks to run
all_tasks="cifar10 imdb"
log_dir=final_exp_logs
models="claude-2.1 gpt-4-0125-preview gemini-pro"

for model in $models
do
    for task in $all_tasks
    do  
        bash multi_run_experiment.sh $log_dir/$model/$task $task 8 {0..7} --llm-name $model --edit-script-llm-name $model --fast-llm-name $model
    done
done

# other agent variants

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/retrieval/$task $task 8 {0..7}  --retrieval 
done

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/autogpt/$task $task 8 {0..7} --agent-type AutoGPTAgent 
done

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/react/$task $task 8 {0..7} --agent-type ReasoningActionAgent 

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/langchain/$task $task 8 {0..7}  --agent-type LangChainAgent 
done
