""" This file contains the agent class for the LangChain agent, which adapts the LangChain agents to the MLAgentBench framework."""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any


from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.chat_models import ChatAnthropic
from langchain.schema import (
    AgentAction,
    AgentFinish
)
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.input import get_color_mapping
from langchain.callbacks import FileCallbackHandler
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from MLAgentBench.schema import Action
from .agent import Agent


class AgentExecutorWithState(AgentExecutor):
    """ A modified version of the AgentExecutor class that allows us to keep track of the agent's state. """

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = inputs["intermediate_steps"]
        # Let's start tracking the number of iterations and time elapsed
        iterations = inputs["iterations"]
        time_elapsed = inputs["time_elapsed"]
        # start_time = time.time()
        if inputs["start_time"] is None:
            start_time = time.time()
        else:
            start_time = inputs["start_time"]
        del inputs["intermediate_steps"], inputs["iterations"], inputs["time_elapsed"], inputs["start_time"]
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )

            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)

            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time

        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)


class AnthropicOutputParser(MRKLOutputParser):
    """ Modified version of the MRKLOutputParser that allows us to parse the output of an anthropic models. """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        text = text.split("Thought:")[-1]
        return super().parse(text)

    @property
    def _type(self) -> str:
        return "anthropic"


class EnvTool:
    """ A wrapper class to wrap actions as tools for the LangChain agent. """
    def __init__(self, action_info, env):
        self.action_info = action_info
        self.env = env

    def run(self, action_input: str) -> str:
        """Run command and returns anything printed."""
        try:
            parsed_input = LangChainAgent.parse_action_input(action_input, self.action_info)
            observation = self.env.execute(Action(self.action_info.name, parsed_input))
        except Exception as e:
            # error parsing
            usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_info.usage.items()])
            usage = f"""{{
    {usage}
}}"""
            invalid_action_error = f"The action input for {self.action_info.name} needs to be a valid json with proper entries. You may have missed the comma between entries. Please use the correct format and try again:\n{usage}"
            observation = "ActionInputParsingError: "+ e + "\n" + invalid_action_error

        return observation


class LangChainAgent(Agent):   
    """ A wrapper class to wrap the LangChain agents to the MLAgentBench framework."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.intermediate_steps = []
        self.iterations = 0
        self.time_elapsed = 0.0
        self.start_time = None

    def run(self, env):

        # init langchain agents
        if self.args.llm_name.startswith("claude"):
            llm = ChatAnthropic(model=self.args.llm_name, anthropic_api_key=open("claude_api_key.txt").read().strip(), temperature=0.5, max_tokens_to_sample = 2000)
            agent_kwargs = {"output_parser": AnthropicOutputParser()}
        else:
            # TODO: add support for other agents
            raise NotImplementedError

        tools = []
        for tool_name in self.prompt_tool_names:
            tools.append(Tool(
                tool_name, 
                EnvTool(self.action_infos[tool_name], env).run,
                self.construct_tool_prompt(tool_name, self.action_infos[tool_name]).replace("{", "{{").replace("}", "}}")
                )
            )

        AgentExecutor._call = AgentExecutorWithState._call 
        agent = initialize_agent(tools, llm, agent=self.args.langchain_agent, max_iterations = self.args.agent_max_steps, return_intermediate_steps=True, agent_kwargs = agent_kwargs, verbose=True)

        with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
            f.write(agent.agent.llm_chain.prompt.template) 


        inputs = {
            "input": env.research_problem,
            "intermediate_steps": self.intermediate_steps,
            "iterations": self.iterations,
            "time_elapsed": self.time_elapsed,
            "start_time": self.start_time,
        }

        log_file = os.path.join(self.log_dir , f"step_log.log")

        #change std out to log file
        sys.stdout = open(os.path.join(self.log_dir, "main_log"), "a", 1)
        finish_state = agent(inputs, callbacks= [FileCallbackHandler(filename=log_file)])
        sys.stdout = sys.__stdout__

        self.save(os.path.join(self.log_dir , f"agent_{self.iterations}.json"))

        return finish_state["output"]
    
