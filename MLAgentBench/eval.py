import os
import anthropic
from pathlib import Path
import re
import sys
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from importlib import util
import argparse
import importlib 
import matplotlib.pyplot as plt

# from .LLM import complete_text_gpt4, complete_text_claude
from .environment import get_task_info



class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        #if it is a function, use its string name
        elif hasattr(o, '__call__'):
            return o.__name__

        return super().default(o)

def oom_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    message = "CUDA out of memory"
    return (message in open(log, "r").read()) or (message in open(main_log, "r").read())
    

def connection_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    bad = ["You exceeded your current quota, please check your plan and billing details.", "Error: 'text-similarity-ada-001'", "Error: 'text-embedding-ada-001'"]
    return ("Connection aborted" in open(log, "r").read()) or (any([b in open(main_log, "r").read() for b in bad])) 

def error(path):
    return os.path.exists(os.path.join(path.replace("trace.json", ""), "error.txt")) or not os.path.exists(os.path.join(path.replace("trace.json", ""), "overall_time.txt"))


def json_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return open(main_log, "r").read().count("JSONDecodeError") > 2

def long_prompt_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return "EnvError: too long input for the tool" in open(main_log, "r").read()

@dataclass
class EvaluationResult:
    path: str
    summary: str
    rubric_questions: Dict[str, str]
    score: List[float]
    score_steps: List[float]
    submitted_final_answer: bool
    final_score: float
    total_time: float
    error: str
    extra: Dict[str, bool]


def run_eval(log_folder, benchmark_folder_name, eval_intermediate=False):
    initial_prompt = '''You are an evaluator and have to reflect on the performance of a research assistant. 
    You shall be given the summarized logs of what all actions the research assistant tried and what were the observations on executing those
    actions. Based on that, answer the following question in only Y or N: \n
    '''
    rubric_questions = [
    "Did the run give a final answer and actually solved the task?", 
    "Did the run give a final answer but due to some hallucination?",
    "Did the run fail in some sort of debugging and hence max steps were reached and agent could not come with a final answer?",
    "Did the run failed due to max token length exceeded?",
    "Did the run show some hallucination in intermediate steps irrespective of the final answer/issue?",
    "Did the agent show lack of a research plan and steps followed by it were not very logical?",
    "Did the agent show some sort of lack of domain expertise in that research area?"]

    final_prompt = '''You are given a summarized log of actions and observations in order to solve a task. 
    Please provide a final 2-3 line summary of what happened in the run for a human to understand the log.'''

    results = {}    
    summarized_log = ''
    for subdir, dirs, files in os.walk(log_folder):
        for file in files:

            if file == 'trace.json':
                result = EvaluationResult(
                    path=os.path.join(subdir, file),
                    summary="",
                    rubric_questions={},
                    score=[],
                    score_steps=[],
                    final_score = -1,
                    submitted_final_answer = False,
                    total_time = 0,
                    error = "",
                    extra = {}
                )
                try:
                    with open(os.path.join(subdir, file)) as f:
                        data = json.load(f)
                except:
                    continue
                num_steps = len(data['steps'])
                for step in range(len(data['steps'])):
                    if data['steps'][step]["action"]["name"] == "Final Answer":
                        result.submitted_final_answer = True
                
                # for step in range(len(data['steps'])):
                #     summarized_log += f"\n Step {step}: \n"
                #     summarized_log += f"Action: {data['steps'][step]['action']['name']} \n"
                #     prompt = f'''You are given the observation on executing an action {data['steps'][step]['action']['name']}. \n 
                #     The arguments for this action are {data['steps'][step]['action']['args']}. \n
                #     This leads to the following observation: \n {data['steps'][step]['observation']}.\n
                #     Based on the above, your task is to crisply summarize this step based on the action taken and the observation received in 3-4 lines.
                #     '''
                #     summarized_observation = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], log_file=None)
                #     summarized_log += f"Summarized Observation: {summarized_observation} \n"
                
                # result.summary = summarized_log

                # for ques in rubric_questions:
                #     prompt = initial_prompt + "\n" + summarized_log + "\n" + ques + "\n"
                #     answer = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], log_file=None)
                #     result.rubric_questions[ques] = answer.strip()

                # prompt = summarized_log + "\n" + final_prompt + "\n"
                # answer = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], log_file=None)
                # result.rubric_questions[ques] = answer.strip()
                # eval_score = 0

                # file_path = os.path.join(subdir, 'traces/step_final_files/submission.csv')
                # if os.path.exists(file_path):
                #     module = importlib.import_module(f'benchmarks.{benchmark_folder_name}.scripts.eval')
                #     eval_score = module.get_score(file_path)
                # else:
                #     file_path = os.path.join(subdir, f'traces/step_{num_steps-1}_files/submission.csv')
                #     if os.path.exists(file_path):
                #         module = importlib.import_module(f'benchmarks.{benchmark_folder_name}.scripts.eval')
                #         eval_score = module.get_score(file_path)
                
                num_steps_eval = 5
                step_list = range(num_steps)
                if num_steps_eval >= len(step_list):
                    subsampled_list = step_list
                else:
                    step = num_steps // num_steps_eval
                    subsampled_list = step_list[::step][:num_steps_eval]

                if eval_intermediate:
                    for step in subsampled_list:
                        eval_step_score = 0
                        try:
                            folder_path = os.path.join(subdir, f'traces/step_{step}_files')
                            if os.path.exists(folder_path):
                                print(folder_path)
                                module = importlib.import_module(f'MLAgentBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                                eval_step_score = module.get_score(folder_path)
                                result.score.append(eval_step_score)
                        except Exception as e:
                            print(e)
                            result.score.append(eval_step_score)
                    result.score_steps = list(subsampled_list)
                            
                folder_path = os.path.join(subdir, 'traces/step_final_files')
                try:
                    if os.path.exists(folder_path):
                        module = importlib.import_module(f'MLAgentBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                        eval_final_score = module.get_score(folder_path)
                        result.score.append(eval_final_score)
                        result.final_score = eval_final_score
                        print(eval_final_score)
                except Exception as e:
                    print(e)
                    pass
                
                
                if os.path.exists(os.path.join(subdir, "error.txt")):
                    result.error = open(os.path.join(subdir, "error.txt")).read()
                
                if os.path.exists(os.path.join(subdir, "overall_time.txt")):
                    result.total_time = float(open(os.path.join(subdir, "overall_time.txt")).read())
                    print(result.total_time)
                
                result.extra = {
                    "oom_error": oom_error(os.path.join(subdir, file)),
                    "connection_error": connection_error(os.path.join(subdir, file)),
                    "error": error(os.path.join(subdir, file)),
                    "json_error": json_error(os.path.join(subdir, file)),
                    "long_prompt_error": long_prompt_error(os.path.join(subdir, file)),
                }
                    
                results[os.path.join(subdir, file)] = result
                    
        
    return results
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-folder", type=str, default="logs")
    parser.add_argument("--task", type=str, default="cifar10_training")
    parser.add_argument("--output-file", type=str, default="results.json")
    parser.add_argument("--eval-intermediate", action="store_true")
    args = parser.parse_args()
    
    benchmark_folder_name = get_task_info(args.task)[0] 
    results = run_eval(args.log_folder, benchmark_folder_name, eval_intermediate = args.eval_intermediate)
              
    json.dump(results, open(args.output_file, "w"), indent=4, cls=EnhancedJSONEncoder)
                
       
