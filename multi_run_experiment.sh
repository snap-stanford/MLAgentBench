#/bin/bash


exp_path=$1
task=$2
n_device=$3
shift 3

declare -a devices=()

# Get X numbers
for (( i=0; i<$n_device; i++ ))
do
  devices+=($1)
  shift
done

extra_args="${@}"
folder=$exp_path
python=$(which python)

echo "exp_path: $exp_path"
echo "task: $task"
echo "n_devices: $n_device"
echo "devcies: ${devices[@]}"
echo "extra_args: $extra_args"

echo "Logs will be saved to $folder"

for i in "${devices[@]}"
do 

    ts=$(date +%s)

    if [ -d $folder/$ts ]; then
        echo "Folder $folder/$ts already exists. removing it"
        rm -rf $folder/$ts
    fi
    mkdir -p $folder/$ts


    python -u -m MLAgentBench.prepare_task $task $python
    echo "python -u -m MLAgentBench.runner --python $python --task $task --device $i --log-dir $folder/$ts  --work-dir workspaces/$folder/$ts ${extra_args} > $folder/$ts/log 2>&1"
    eval "python -u -m MLAgentBench.runner --python $python --task $task --device $i --log-dir $folder/$ts  --work-dir workspaces/$folder/$ts ${extra_args}" > $folder/$ts/log 2>&1 &


    sleep 2
done
wait

