import json
from typing import Optional
from funkagent import parser
import os

import openai

sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussion on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""


class Agent:
    def __init__(
        self,
        openai_api_key: Optional[str] = os.environ.get('AZURE_OPENAI_API_KEY'),
        openai_api_base: Optional[str] = os.environ.get('AZURE_OPENAI_BASE'),
        openai_api_version: Optional[str] = os.environ.get('AZURE_OPENAI_API_VERSION'),
        deployment_name: Optional[str] = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT'),
        functions: Optional[list] = None
    ):
        openai.api_type = "azure"
        openai.api_key = openai_api_key
        openai.api_base = openai_api_base
        openai.api_version = openai_api_version        

        self.deployment_name = deployment_name
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        self.chat_history = [{'role': 'system', 'content': sys_msg}]

    def _parse_functions(self, functions: Optional[list]) -> Optional[list]:
        if functions is None:
            return None
        return [parser.func_to_json(func) for func in functions]

    def _create_func_mapping(self, functions: Optional[list]) -> dict:
        if functions is None:
            return {}
        return {func.__name__: func for func in functions}

    async def _create_chat_completion(
        self, messages: list, use_functions: bool=True
    ) -> openai.ChatCompletion:
        if use_functions and self.functions:
            res = await openai.ChatCompletion.acreate(
                deployment_id=self.deployment_name,
                messages=messages,
                functions=self.functions,
                function_call="auto"
            )
        else:
            res = await openai.ChatCompletion.acreate(
                deployment_id=self.deployment_name,
                messages=messages
            )
        return res

    async def _generate_response(self) -> openai.ChatCompletion:
        while True:
            print('.', end='')
            res = await self._create_chat_completion(
                self.chat_history + self.internal_thoughts
            )
            finish_reason = res.choices[0].finish_reason

            if finish_reason == 'stop' or len(self.internal_thoughts) > 3:
                # create the final answer
                final_thought = self._final_thought_answer()
                final_res = await self._create_chat_completion(
                    self.chat_history + self.internal_thoughts + [final_thought],
                    use_functions=False
                )
                return final_res
            elif finish_reason == 'function_call':
                self._handle_function_call(res)
            else:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")

    def _handle_function_call(self, res: openai.ChatCompletion):
        res_func_call = res.choices[0].message.to_dict()
        res_func_call['content'] = None # content key is required
        self.internal_thoughts.append(res_func_call)
        func_name = res.choices[0].message.function_call.name
        args_str = res.choices[0].message.function_call.arguments
        result = self._call_function(func_name, args_str)
        res_msg = {'role': 'function', 'name': func_name, 'content': str(result)}
        self.internal_thoughts.append(res_msg)

    def _call_function(self, func_name: str, args_str: str):
        try:
            args = json.loads(args_str)
            func = self.func_mapping[func_name]
            res = func(**args)
        except Exception as e:
            res = f"Exception: {e}"
        return res
    
    def _final_thought_answer(self):

        return {
            "role" : "assistant", 
            "content": "The user has only seen the last user input. Consider this, when providing your response."
        }
    
        # thoughts = ("To answer the question I will use these step by step instructions."
        #             "\n\n")
        # for thought in self.internal_thoughts:
        #     if 'function_call' in thought.keys():
        #         thoughts += (f"I will use the {thought['function_call']['name']} "
        #                      "function to calculate the answer with arguments "
        #                      + thought['function_call']['arguments'] + ".\n\n")
        #     else:
        #         thoughts += thought["content"] + "\n\n"
        # self.final_thought = {
        #     'role': 'assistant',
        #     'content': (f"{thoughts} Based on the above, I will now answer the "
        #                 "question, this message will only be seen by me so answer with "
        #                 "the assumption with that the user has not seen this message.")
        # }
        # return self.final_thought

    async def ask(self, query: str) -> openai.ChatCompletion:
        self.internal_thoughts = []
        self.chat_history.append({'role': 'user', 'content': query})
        res = await self._generate_response()
        self.chat_history.append(res.choices[0].message.to_dict())
        return res.choices[0].message.content