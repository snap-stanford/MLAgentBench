

all_tasks="cifar10 imdb"
models="no_retrieval_claude2.1 no_retrieval_gpt-4-0125-preview no_retrieval_gemini_pro"

for model in $models
do
    for task in $all_tasks
    do
        echo "python -m MLAgentBench.eval --log-folder final_exp_logs/$model/$task --task $task --output-file ${model}_${task}.json"
        python -m MLAgentBench.eval --log-folder final_exp_logs/$model/$task --task $task --output-file ${model}_${task}.json 
        # add --eval-intermediate to evaluate intermediate steps 
    done
done