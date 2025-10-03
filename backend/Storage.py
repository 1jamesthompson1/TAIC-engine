"""
Azure Blob Storage implementation for storing conversation histories and knowledge search logs.
This provides a base class for blob storage operations and specialized classes for different data types.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from azure.storage.blob import BlobServiceClient
from abc import ABC, abstractmethod
import os
from . import Version


class BaseBlobStore(ABC):
    """
    Base class for Azure Blob Storage operations.
    Provides common functionality for storing and retrieving JSON data as blobs.
    """
    
    def __init__(self, connection_string: str, container_name: str):
        """
        Initialize the Blob Storage client.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to store blobs
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.container_name = container_name
        
        # Initialize Blob client
        self.blob_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Create container if it doesn't exist
        self._setup_container()
    
    def _get_connection_string(self) -> str:
        """
        Build connection string from environment variables.
        Uses the same storage account as your existing Table Storage.
        """
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        
        if not account_name or not account_key:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set")
        
        return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    
    def _setup_container(self):
        """Create blob container if it doesn't exist."""
        try:
            self.container_client = self.blob_client.get_container_client(self.container_name)
            
            # Create container if it doesn't exist
            if not self.container_client.exists():
                self.container_client.create_container()
                print(f"Created blob container: {self.container_name}")
            
        except Exception as e:
            raise Exception(f"Failed to setup blob container: {e}")
    
    @abstractmethod
    def _get_blob_name(self, *args, **kwargs) -> str:
        """Generate blob name for the specific data type. Must be implemented by subclasses."""
        pass
    
    def store_blob(self, blob_name: str, data: Union[Dict, List], content_type: str = "application/json") -> Optional[str]:
        """
        Store data as a JSON blob.
        
        Args:
            blob_name: Name/path for the blob
            data: Data to store (will be JSON serialized)
            content_type: MIME type for the blob content
            
        Returns:
            Blob name/path for the stored data
        """
        if not data:
            return None
            
        try:
            # Convert data to JSON
            json_data = json.dumps(data, ensure_ascii=False, indent=None)
            
            # Upload to blob storage
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                json_data, 
                content_type=content_type,
                overwrite=True,
                encoding='utf-8'
            )
            
            print(f"✓ Stored blob: {blob_name} ({len(json_data)} bytes)")
            return blob_name
            
        except Exception as e:
            print(f"✗ Failed to store blob {blob_name}: {e}")
            # Save failed data to local file for debugging
            self._save_failed_data(blob_name, data, str(e))
            raise Exception(f"Failed to store blob: {e}")
    
    def retrieve_blob(self, blob_name: str) -> Optional[Union[Dict, List]]:
        """
        Retrieve data from blob storage.
        
        Args:
            blob_name: Name/path of the blob to retrieve
            
        Returns:
            Parsed JSON data or None if not found
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download blob content
            blob_data = blob_client.download_blob()
            json_content = blob_data.readall().decode('utf-8')
            
            # Parse JSON
            data = json.loads(json_content)
            print(f"✓ Retrieved blob: {blob_name}")
            return data
            
        except Exception as e:
            print(f"✗ Failed to retrieve blob {blob_name}: {e}")
            return None
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob.
        
        Args:
            blob_name: Name/path of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            print(f"✓ Deleted blob: {blob_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete blob {blob_name}: {e}")
            return False
    
    def _save_failed_data(self, blob_name: str, data: Union[Dict, List], error: str):
        """Save failed data to file for debugging."""
        try:
            # Extract a meaningful part of the blob name for the filename
            safe_name = blob_name.replace("/", "_").replace("\\", "_")
            filename = f"Failed_blob_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding='utf-8') as f:
                json.dump({
                    "error": error,
                    "blob_name": blob_name,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved failed data to {filename}")
        except Exception as e:
            print(f"Could not save failed data: {e}")


class ConversationBlobStore(BaseBlobStore):
    """
    Handles conversation storage using Azure Blob Storage for JSON data.
    Inherits from BaseBlobStore for common blob operations.
    """
    
    def _get_blob_name(self, username: str, conversation_id: str) -> str:
        """Generate blob name for a conversation."""
        return f"{username}/{conversation_id}.json"
    
    def store_conversation_blob(self, username: str, conversation_id: str, history: List[Dict]) -> Optional[str]:
        """
        Store conversation history as a JSON blob.
        
        Args:
            username: Username of the conversation owner
            conversation_id: Unique identifier for the conversation
            history: List of conversation messages
            
        Returns:
            Blob name/path for the stored conversation or None if failed
        """
        if not history:
            return None
            
        blob_name = self._get_blob_name(username, conversation_id)
        return self.store_blob(blob_name, history)
    
    def retrieve_conversation_blob(self, username: str, conversation_id: str) -> Optional[List[Dict]]:
        """
        Retrieve conversation history from blob storage.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            List of conversation messages or None if not found
        """
        blob_name = self._get_blob_name(username, conversation_id)
        data = self.retrieve_blob(blob_name)
        # Ensure we return the correct type - conversations should be lists
        if isinstance(data, list):
            return data
        return None
    
    def delete_conversation_blob(self, username: str, conversation_id: str) -> bool:
        """
        Delete a conversation blob.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        blob_name = self._get_blob_name(username, conversation_id)
        return self.delete_blob(blob_name)


class KnowledgeSearchBlobStore(BaseBlobStore):
    """
    Handles knowledge search log storage using Azure Blob Storage for detailed search data.
    Stores comprehensive search results and parameters that exceed Table Storage limits.
    """
    
    def _get_blob_name(self, username: str, search_id: str) -> str:
        """Generate blob name for a knowledge search log."""
        return f"searches/{username}/{search_id}.json"
    
    def store_search_blob(self, username: str, search_id: str, search_data: Dict) -> Optional[str]:
        """
        Store knowledge search data as a JSON blob.
        
        Args:
            username: Username of the search owner
            search_id: Unique identifier for the search
            search_data: Complete search data including results, settings, and metadata
            
        Returns:
            Blob name/path for the stored search data or None if failed
        """
        if not search_data:
            return None
            
        blob_name = self._get_blob_name(username, search_id)
        return self.store_blob(blob_name, search_data)
    
    def retrieve_search_blob(self, username: str, search_id: str) -> Optional[Dict]:
        """
        Retrieve knowledge search data from blob storage.
        
        Args:
            username: Username of the search owner
            search_id: ID of the search to retrieve
            
        Returns:
            Search data dictionary or None if not found
        """
        blob_name = self._get_blob_name(username, search_id)
        data = self.retrieve_blob(blob_name)
        # Ensure we return the correct type - search data should be dicts
        if isinstance(data, dict):
            return data
        return None
    
    def delete_search_blob(self, username: str, search_id: str) -> bool:
        """
        Delete a knowledge search blob.
        
        Args:
            username: Username of the search owner
            search_id: ID of the search to delete
            
        Returns:
            True if successful, False otherwise
        """
        blob_name = self._get_blob_name(username, search_id)
        return self.delete_blob(blob_name)


class ConversationMetadataStore:
    """
    Handles conversation metadata storage in Table Storage.
    This works alongside the blob storage for complete conversation management.
    """
    
    def __init__(self, table_client, blob_store: ConversationBlobStore):
        """
        Initialize with existing table client and blob store.
        
        Args:
            table_client: Azure Table Storage client (reuse existing)
            blob_store: ConversationBlobStore instance
        """
        self.table_client = table_client
        self.blob_store = blob_store
    
    def create_or_update_conversation(
        self, username: str, conversation_id: str, db_version: int,
        history: List[Dict], conversation_title: Optional[str] = None,
        ) -> bool:
        """
        Store conversation: blob for JSON data, Table Storage for metadata.
        
        Args:
            username: Username of the conversation owner
            conversation_id: Unique identifier for the conversation
            history: List of conversation messages
            conversation_title: Title for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        if not history:
            return False
            
        try:
            # 1. Store conversation JSON in blob storage
            blob_name = self.blob_store.store_conversation_blob(username, conversation_id, history)
            if not blob_name:
                return False
            
            # 2. Store metadata in Table Storage
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if conversation already exists to preserve created_at
            existing_entity = None
            try:
                existing_entity = self.table_client.get_entity(
                    partition_key=username, 
                    row_key=conversation_id
                )
            except:
                pass  # Entity doesn't exist, which is fine
            
            entity = {
                "PartitionKey": username,
                "RowKey": conversation_id,
                "conversation_title": conversation_title or "Untitled Conversation",
                "blob_name": blob_name,
                "message_count": len(history),
                "last_updated": now,
                "created_at": existing_entity.get("created_at", now) if existing_entity else now,
                "app_version": Version.CURRENT_VERSION,
                "db_version": db_version,
            }
            
            # Upsert the entity (create or update)
            self.table_client.upsert_entity(entity=entity)
            
            print(f"✓ Stored conversation metadata for {conversation_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store conversation {conversation_id}: {e}")
            return False
    
    def get_user_conversations_metadata(self, username: str) -> List[Dict]:
        """
        Retrieve only conversation metadata for a user (no full message history).
        
        Args:
            username: Username to retrieve conversations for
            
        Returns:
            List of conversation metadata without full message history
        """
        try:
            # Get metadata from Table Storage only
            entities = self.table_client.query_entities(
                query_filter=f"PartitionKey eq '{username}'"
            )
            
            conversations = []
            
            for entity in entities:
                conversations.append({
                    "conversation_title": entity.get("conversation_title"),
                    "id": entity.get("RowKey"),
                    "last_updated": entity.get("last_updated"),
                    "created_at": entity.get("created_at"),
                    "message_count": entity.get("message_count", 0),
                    "blob_name": entity.get("blob_name"),
                    "app_version": entity.get("app_version"),
                })
            
            # Sort by last updated
            conversations.sort(
                key=lambda x: datetime.strptime(x["last_updated"], "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            
            print(f"✓ Retrieved metadata for {len(conversations)} conversations for {username}")
            return conversations
            
        except Exception as e:
            print(f"✗ Failed to retrieve conversation metadata for {username}: {e}")
            return []

    def load_single_conversation(self, username: str, conversation_id: str) -> Optional[Dict]:
        """
        Load a single conversation with full message history.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to load
            
        Returns:
            Dictionary with conversation data including full message history, or None if not found
        """
        try:
            # Get metadata from Table Storage
            entity = self.table_client.get_entity(
                partition_key=username, 
                row_key=conversation_id
            )
            
            # Get full conversation history from blob storage
            history = self.blob_store.retrieve_conversation_blob(username, conversation_id)
            if history is None:
                print(f"⚠ Could not retrieve blob for conversation {conversation_id}")
                return None
            
            conversation = {
                "conversation_title": entity.get("conversation_title"),
                "messages": history,
                "id": conversation_id,
                "last_updated": entity.get("last_updated"),
                "created_at": entity.get("created_at"),
                "message_count": entity.get("message_count", len(history)),
                "app_version": entity.get("app_version"),
            }
            
            return conversation
            
        except Exception as e:
            print(f"✗ Failed to load conversation {conversation_id} for {username}: {e}")
            return None
    
    def delete_conversation(self, username: str, conversation_id: str) -> bool:
        """
        Delete both metadata and blob for a conversation.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete blob first
            self.blob_store.delete_conversation_blob(username, conversation_id)
            
            # Delete metadata from table
            self.table_client.delete_entity(
                partition_key=username,
                row_key=conversation_id
            )
            
            print(f"✓ Deleted conversation {conversation_id} for {username}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete conversation {conversation_id}: {e}")
            return False


class KnowledgeSearchMetadataStore:
    """
    Handles knowledge search metadata storage in Table Storage.
    Works alongside KnowledgeSearchBlobStore for complete search log management.
    """
    
    def __init__(self, table_client, blob_store: KnowledgeSearchBlobStore):
        """
        Initialize with existing table client and blob store.
        
        Args:
            table_client: Azure Table Storage client (reuse existing)
            blob_store: KnowledgeSearchBlobStore instance
        """
        self.table_client = table_client
        self.blob_store = blob_store
    
    def store_search_log(self, username: str, search_id: str, search_settings: Dict, 
                        results_info: Dict, error_info: Optional[Dict] = None) -> bool:
        """
        Store knowledge search: blob for detailed data, Table Storage for metadata.
        
        Args:
            username: Username of the search owner
            search_id: Unique identifier for the search
            search_settings: Search parameters and settings
            results_info: Information about search results (count, relevance, etc.)
            error_info: Error information if search failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Prepare comprehensive data for blob storage
            blob_data = {
                "search_settings": search_settings,
                "results_info": results_info,
                "error_info": error_info,
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "search_id": search_id
            }
            
            # Store detailed data in blob storage
            blob_name = self.blob_store.store_search_blob(username, search_id, blob_data)
            if not blob_name:
                return False
            
            # 2. Store lightweight metadata in Table Storage
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            entity = {
                "PartitionKey": username,
                "RowKey": search_id,
                "query": search_settings.get("query", "")[:100],  # Truncate for table storage
                "search_type": search_settings.get("type", "unknown"),
                "document_types": str(search_settings.get("document_type", [])),
                "modes": str(search_settings.get("modes", [])),
                "agencies": str(search_settings.get("agencies", [])),
                "relevance_threshold": search_settings.get("relevance", 0.0),
                "total_results": results_info.get("total_results", 0),
                "relevant_results": results_info.get("relevant_results", 0),
                "has_error": error_info is not None,
                "error_message": str(error_info.get("error", ""))[:200] if error_info else "",
                "blob_name": blob_name,
                "search_timestamp": now,
                "created_at": now,
                "app_version": Version.CURRENT_VERSION,
            }
            
            # Store metadata
            self.table_client.create_entity(entity=entity)
            
            print(f"✓ Stored search log metadata for {search_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store search log {search_id}: {e}")
            return False
    
    def get_user_search_history(self, username: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve search history metadata for a user (without full detailed data).
        
        Args:
            username: Username to retrieve search history for
            limit: Optional limit on number of results
            
        Returns:
            List of search metadata without full detailed data
        """
        try:
            # Get metadata from Table Storage only
            entities = self.table_client.query_entities(
                query_filter=f"PartitionKey eq '{username}'"
            )
            
            searches = []
            
            for entity in entities:
                searches.append({
                    "search_id": entity.get("RowKey"),
                    "query": entity.get("query"),
                    "search_type": entity.get("search_type"),
                    "total_results": entity.get("total_results", 0),
                    "relevant_results": entity.get("relevant_results", 0),
                    "has_error": entity.get("has_error", False),
                    "error_message": entity.get("error_message"),
                    "search_timestamp": entity.get("search_timestamp"),
                    "created_at": entity.get("created_at"),
                    "blob_name": entity.get("blob_name"),
                    "app_version": entity.get("app_version"),
                })
            
            # Sort by search timestamp (most recent first)
            searches.sort(
                key=lambda x: datetime.strptime(x["search_timestamp"], "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            
            # Apply limit if specified
            if limit:
                searches = searches[:limit]
            
            print(f"✓ Retrieved {len(searches)} search history records for {username}")
            return searches
            
        except Exception as e:
            print(f"✗ Failed to retrieve search history for {username}: {e}")
            return []

    def load_detailed_search(self, username: str, search_id: str) -> Optional[Dict]:
        """
        Load a single search with full detailed data from blob storage.
        
        Args:
            username: Username of the search owner
            search_id: ID of the search to load
            
        Returns:
            Dictionary with complete search data including detailed results, or None if not found
        """
        try:
            # Get metadata from Table Storage
            entity = self.table_client.get_entity(
                partition_key=username, 
                row_key=search_id
            )
            
            # Get full detailed data from blob storage
            detailed_data = self.blob_store.retrieve_search_blob(username, search_id)
            if detailed_data is None:
                print(f"⚠ Could not retrieve blob for search {search_id}")
                return None
            
            # Combine metadata and detailed data
            search_data = {
                "search_id": search_id,
                "metadata": {
                    "query": entity.get("query"),
                    "search_type": entity.get("search_type"),
                    "total_results": entity.get("total_results", 0),
                    "relevant_results": entity.get("relevant_results", 0),
                    "has_error": entity.get("has_error", False),
                    "error_message": entity.get("error_message"),
                    "search_timestamp": entity.get("search_timestamp"),
                    "created_at": entity.get("created_at"),
                },
                "detailed_data": detailed_data
            }
            
            return search_data
            
        except Exception as e:
            print(f"✗ Failed to load detailed search {search_id} for {username}: {e}")
            return None
    
    def delete_search_log(self, username: str, search_id: str) -> bool:
        """
        Delete both metadata and blob for a search log.
        
        Args:
            username: Username of the search owner
            search_id: ID of the search to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete blob first
            self.blob_store.delete_search_blob(username, search_id)
            
            # Delete metadata from table
            self.table_client.delete_entity(
                partition_key=username,
                row_key=search_id
            )
            
            print(f"✓ Deleted search log {search_id} for {username}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete search log {search_id}: {e}")
            return False
