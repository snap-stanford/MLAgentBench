#/bin/bash

# example tasks to run
all_tasks="cifar10 imdb"
log_dir=final_exp_logs

#### gpt4 experiments with pure openai API 
for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/gpt4/$task $task 8 {0..7} --llm-name gpt-4 --max-steps 30 --edit-script-llm-name gpt-4 --fast-llm-name gpt-3.5-turbo
done

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/no_retrieval_gpt4/$task $task 8 {0..7} --llm-name gpt-4 --max-steps 30 --edit-script-llm-name gpt-4 --no-retrieval --fast-llm-name gpt-3.5-turbo
done

#### gpt4 experiments with crfm API + claude API as the fast model
for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/gpt4/$task $task 8 {0..7} --llm-name openai/gpt-4-0314 --max-steps 30 --edit-script-llm-name openai/gpt-4-0314
done

for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/no_retrieval_gpt4/$task $task 8 {0..7} --llm-name openai/gpt-4-0314 --max-steps 30 --edit-script-llm-name openai/gpt-4-0314 --no-retrieval 
done

#### Claude experiments

for task in $all_tasks
do    
    bash multi_run_experiment.sh $log_dir/claude/$task $task 8 {0..7} 
done


for task in $all_tasks
do 
    bash multi_run_experiment.sh $log_dir/no_retrieval/$task $task 8 {0..7}  --no-retrieval 
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

