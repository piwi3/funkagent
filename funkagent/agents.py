import json
from typing import Optional
from funkagent import parser
import os
import inspect
import openai
import logging

from history.cosmosdbservice import CosmosConversationClient

from azure.identity import DefaultAzureCredential


sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussion on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""


class Agent:
    def __init__(
        self,
        user_id: str = None,
        openai_api_key: Optional[str] = os.environ.get('AZURE_OPENAI_API_KEY'),
        openai_api_base: Optional[str] = os.environ.get('AZURE_OPENAI_BASE'),
        openai_api_version: Optional[str] = os.environ.get('AZURE_OPENAI_API_VERSION'),
        deployment_name: Optional[str] = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT'),
        functions: Optional[list] = None,
        conversation_id: Optional[int] = None,
    ):
        # Check user_id
        if user_id is None:
            raise ValueError("user_id is required")
        elif not isinstance(user_id, str):
            raise TypeError("user_id must be a string")
        self.user_id = user_id

        # OpenAI Integration Settings
        openai.api_type = "azure"
        openai.api_key = openai_api_key
        openai.api_base = openai_api_base
        openai.api_version = openai_api_version  
        self.deployment_name = deployment_name      

        # Settings for OpenAI Function Calls
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        
        # CosmosDB Integration Settings
        self.AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
        self.AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
        self.AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
        self.AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")

        # Initialize CosmosDB client
        self.cosmos_conversation_client = self.get_cosmos_conversation_client()

        # Get or create conversation in / from CosmosDB
        if conversation_id is not None:
            conversation_dict = self.cosmos_conversation_client.get_conversation(self.user_id, conversation_id)
            self.conversation_id = conversation_dict.get['id'] if conversation_dict else None
        if conversation_id is None:
            conversation_dict = self.cosmos_conversation_client.create_conversation(self.user_id)
            self.conversation_id = conversation_dict['id']

        # Get chat history from CosmosDB
        self.chat_history = [
            {
                "role": message["role"],
                "content": message["content"]
            } 
            for message in self.cosmos_conversation_client.get_messages(self.user_id, self.conversation_id)
        ]

        # Add system message if chat history is empty
        if self.chat_history == []:
            message = {'role': 'system', 'content': sys_msg}
            self.chat_history.append(message)
            self.cosmos_conversation_client.create_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    input_message=message
                )

    def get_cosmos_conversation_client(self) -> CosmosConversationClient:
        # Initialize a CosmosDB client with AAD auth and containers
        if self.AZURE_COSMOSDB_DATABASE and self.AZURE_COSMOSDB_ACCOUNT and self.AZURE_COSMOSDB_CONVERSATIONS_CONTAINER:
            try :
                cosmos_endpoint = f'https://{self.AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/'

                if not self.AZURE_COSMOSDB_ACCOUNT_KEY:
                    credential = DefaultAzureCredential()
                else:
                    credential = self.AZURE_COSMOSDB_ACCOUNT_KEY

                return CosmosConversationClient(
                    cosmosdb_endpoint=cosmos_endpoint, 
                    credential=credential, 
                    database_name=self.AZURE_COSMOSDB_DATABASE,
                    container_name=self.AZURE_COSMOSDB_CONVERSATIONS_CONTAINER
                )
            except Exception as e:
                logging.exception("Exception in CosmosDB initialization", e)
                return None

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
                func_name = res.choices[0].message.function_call.name
                if inspect.iscoroutinefunction(self.func_mapping[func_name]):
                    await self._ahandle_function_call(res)
                else:
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

    async def _ahandle_function_call(self, res: openai.ChatCompletion):
        res_func_call = res.choices[0].message.to_dict()
        res_func_call['content'] = None # content key is required
        self.internal_thoughts.append(res_func_call)
        func_name = res.choices[0].message.function_call.name
        args_str = res.choices[0].message.function_call.arguments
        result = await self._acall_function(func_name, args_str)
        res_msg = {'role': 'function', 'name': func_name, 'content': str(result)}
        self.internal_thoughts.append(res_msg)
    
    async def _acall_function(self, func_name: str, args_str: str):
        try:
            args = json.loads(args_str)
            func = self.func_mapping[func_name]
            res = await func(**args)
        except Exception as e:
            res = f"Exception: {e}"
        return res
    
    def _final_thought_answer(self):

        return {
            "role" : "assistant", 
            "content": "The user has only seen the last user input. Consider this, when providing your response."
        }
    
    async def ask(self, query: str) -> openai.ChatCompletion:

        message_user = {'role': 'user', 'content': query}
        self.chat_history.append(message_user)
        self.cosmos_conversation_client.create_message(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            input_message=message_user
        )

        self.internal_thoughts = []
        res = await self._generate_response()

        message_assistant = res.choices[0].message.to_dict()
        self.chat_history.append(message_assistant)
        self.cosmos_conversation_client.create_message(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            input_message=message_assistant
        )

        return res.choices[0].message.content
    
    def get_conversations(self):
        return self.cosmos_conversation_client.get_conversations(self.user_id)
    
    def delete_conversation(self, conversation_id: Optional[str]=None):
        conversation_id = self.conversation_id if conversation_id is None else conversation_id
        if conversation_id != self.conversation_id:
            # Delete conversation with given id
            return self.cosmos_conversation_client.delete_conversation(self.user_id, conversation_id)
        else:
            # Delete current conversation and chat history
            self.chat_history = []
            res = self.cosmos_conversation_client.delete_conversation(self.user_id, conversation_id)

            # Create new conversation
            conversation_dict = self.cosmos_conversation_client.create_conversation(self.user_id)
            self.conversation_id = conversation_dict['id']
            message = {'role': 'system', 'content': sys_msg}
            self.chat_history.append(message)
            self.cosmos_conversation_client.create_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    input_message=message
                )
            
            # Return result of deletion
            return res