import os
import uuid
from datetime import datetime
from flask import Flask, request
from azure.identity import DefaultAzureCredential 
from azure.cosmos.aio import CosmosClient  
from azure.cosmos import PartitionKey  
import asyncio
  
class CosmosConversationClient():
    
    def __init__(self, cosmosdb_endpoint: str, credential: any, database_name: str, container_name: str):
        self.cosmosdb_endpoint = cosmosdb_endpoint
        self.credential = credential
        self.database_name = database_name
        self.container_name = container_name
        self.cosmosdb_client = CosmosClient(self.cosmosdb_endpoint, credential=credential)

    def create_cosmos_client(self):
        return CosmosClient(self.cosmosdb_endpoint, credential=self.credential)

    async def ensure(self):
        try:
            if not self.cosmosdb_client or not self.database_client or not self.container_client:
                return False
            
            async with self.create_cosmos_client() as client:
                database = client.get_database_client(self.database_name)
                container = database.get_container_client(self.container_name)
                container_info = await container.read()
            if not container_info:
                return False
            
            return True
        except:
            return False

    async def create_conversation(self, user_id, title = ''):
        conversation = {
            'id': str(uuid.uuid4()),  
            'type': 'conversation',
            'createdAt': datetime.utcnow().isoformat(),  
            'updatedAt': datetime.utcnow().isoformat(),  
            'userId': user_id,
            'title': title
        }
        ## TODO: add some error handling based on the output of the upsert_item call
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            resp = await container.upsert_item(conversation)  
        if resp:
            return resp
        else:
            return False
    
    async def upsert_conversation(self, conversation):
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            resp = await container.upsert_item(conversation)
        if resp:
            return resp
        else:
            return False

    async def delete_conversation(self, user_id, conversation_id):
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            conversation = await container.read_item(item=conversation_id, partition_key=user_id)        
        if conversation:
            async with self.create_cosmos_client() as client:
                database = client.get_database_client(self.database_name)
                container = database.get_container_client(self.container_name)
                resp = await container.delete_item(item=conversation_id, partition_key=user_id)
            return resp
        else:
            return True

        
    async def delete_messages(self, conversation_id, user_id):
        ## get a list of all the messages in the conversation
        messages = await self.get_messages(user_id, conversation_id)
        response_list = []
        if messages:
            async with self.create_cosmos_client() as client:
                database = client.get_database_client(self.database_name)
                container = database.get_container_client(self.container_name)
                for message in messages:
                    resp = await container.delete_item(item=message['id'], partition_key=user_id)
                    response_list.append(resp)
            return response_list


    async def get_conversations(self, user_id, sort_order = 'DESC'):
        parameters = [
            {
                'name': '@userId',
                'value': user_id
            }
        ]
        query = f"SELECT * FROM c where c.userId = @userId and c.type='conversation' order by c.updatedAt {sort_order}"
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            results =  container.query_items(query=query, parameters=parameters)

            conversations = []
            async for conversation in results:
                conversations.append(conversation)

        ## if no conversations are found, return None
        if len(conversations) == 0:
            return []
        else:
            return conversations

    async def get_conversation(self, user_id, conversation_id):
        parameters = [
            {
                'name': '@conversationId',
                'value': conversation_id
            },
            {
                'name': '@userId',
                'value': user_id
            }
        ]
        query = f"SELECT * FROM c where c.id = @conversationId and c.type='conversation' and c.userId = @userId"
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            results = container.query_items(query=query, parameters=parameters)
            conversations = []
            async for conversation in results:
                conversations.append(conversation) 
            
        ## if no conversations are found, return None
        if len(conversations) == 0:
            return None
        else:
            return conversations[0]
 
    async def create_message(self, conversation_id, user_id, input_message: dict):
        message = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'userId' : user_id,
            'createdAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat(),
            'conversationId' : conversation_id,
            'role': input_message['role'],
            'content': input_message['content']
        }
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            resp = await container.upsert_item(message)  
        if resp:
            ## update the parent conversations's updatedAt field with the current message's createdAt datetime value
            conversation = await self.get_conversation(user_id, conversation_id)
            conversation['updatedAt'] = message['createdAt']
            await self.upsert_conversation(conversation)
            return resp
        else:
            return False
    


    async def get_messages(self, user_id, conversation_id):
        parameters = [
            {
                'name': '@conversationId',
                'value': conversation_id
            },
            {
                'name': '@userId',
                'value': user_id
            }
        ]
        query = f"SELECT * FROM c WHERE c.conversationId = @conversationId AND c.type='message' AND c.userId = @userId ORDER BY c.timestamp ASC"
        async with self.create_cosmos_client() as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            results = container.query_items(query=query, parameters=parameters)
        
            messages = []
            async for message in results:
                messages.append(message)
           
        ## if no messages are found, return false
        if len(messages) == 0:
            return []
        else:
            return messages

