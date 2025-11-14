"""
Data Fetcher module for Aurora API
Simplified version for fetching member messages
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import urljoin

from config import get_settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and processes member messages from Aurora API"""
    
    def __init__(self, request_delay: float = 5.0):
        """
        Initialize DataFetcher
        
        Args:
            request_delay: Delay in seconds between API requests (default 5.0s)
        """
        self.settings = get_settings()
        self.base_url = self.settings.AURORA_API_URL
        self.messages_endpoint = "/messages/"  # Trailing slash is required per API docs
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session = None
        self.request_delay = request_delay
        
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=False)
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_messages_page(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch a single page of messages from Aurora API
        
        Args:
            skip: Number of records to skip
            limit: Number of records to fetch (max 100)
        
        Returns:
            API response as dictionary
        """
        await self._ensure_session()
        
        url = urljoin(self.base_url, self.messages_endpoint)
        params = {
            "skip": skip,
            "limit": limit
        }
        
        try:
            logger.info(f"Fetching messages page: skip={skip}, limit={limit}")
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                messages = data.get('items', [])
                logger.info(f"Fetched {len(messages)} messages")
                
                normalized_data = {
                    'messages': messages,
                    'total': data.get('total', len(messages))
                }
                return normalized_data
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching messages: {e}")
            raise Exception(f"Failed to fetch messages from Aurora API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    async def fetch_all_messages(self) -> List[Dict[str, Any]]:
        """
        Fetch all messages from Aurora API with pagination
        
        Returns:
            List of all member messages (normalized format)
        """
        all_messages = []
        skip = 0
        limit = 100
        total_fetched = 0
        
        try:
            while True:
                retry_count = 0
                page_fetched = False
                
                while retry_count <= self.max_retries and not page_fetched:
                    try:
                        response = await self.fetch_messages_page(skip=skip, limit=limit)
                        
                        messages = response.get("messages", [])
                        total = response.get("total", 0)
                        
                        if not messages:
                            return normalized_messages if all_messages else []
                        
                        all_messages.extend(messages)
                        total_fetched += len(messages)
                        page_fetched = True
                        
                        logger.info(f"Progress: {total_fetched}/{total} messages fetched")
                        
                        if total_fetched >= total or len(messages) < limit:
                            logger.info(f"Successfully fetched {len(all_messages)} total messages")
                            
                            normalized_messages = []
                            for msg in all_messages:
                                normalized_msg = {
                                    'id': msg.get('id'),
                                    'member_name': msg.get('user_name'),
                                    'content': msg.get('message'),
                                    'timestamp': msg.get('timestamp'),
                                    'user_id': msg.get('user_id')
                                }
                                normalized_messages.append(normalized_msg)
                            
                            return normalized_messages
                        
                        skip += limit
                        await asyncio.sleep(self.request_delay)
                        
                    except Exception as fetch_error:
                        error_msg = str(fetch_error)
                        retry_count += 1
                        
                        if any(code in error_msg for code in ["401", "402", "403", "404", "429"]):
                            if retry_count <= self.max_retries:
                                logger.warning(f"API error (attempt {retry_count}/{self.max_retries}): {error_msg}. Retrying in {self.retry_delay} seconds...")
                                await asyncio.sleep(self.retry_delay)
                            else:
                                logger.warning(f"Max retries reached. Using {total_fetched} messages for processing.")
                                if all_messages:
                                    normalized_messages = []
                                    for msg in all_messages:
                                        normalized_msg = {
                                            'id': msg.get('id'),
                                            'member_name': msg.get('user_name'),
                                            'content': msg.get('message'),
                                            'timestamp': msg.get('timestamp'),
                                            'user_id': msg.get('user_id')
                                        }
                                        normalized_messages.append(normalized_msg)
                                    return normalized_messages
                                else:
                                    raise Exception("Failed to fetch any messages after retries")
                        else:
                            if retry_count <= self.max_retries:
                                logger.warning(f"Unexpected error (attempt {retry_count}/{self.max_retries}): {error_msg}. Retrying in {self.retry_delay} seconds...")
                                await asyncio.sleep(self.retry_delay)
                            else:
                                raise
            
            logger.info(f"Successfully fetched {len(all_messages)} total messages")
            
            normalized_messages = []
            for msg in all_messages:
                normalized_msg = {
                    'id': msg.get('id'),
                    'member_name': msg.get('user_name'),
                    'content': msg.get('message'),
                    'timestamp': msg.get('timestamp'),
                    'user_id': msg.get('user_id')
                }
                normalized_messages.append(normalized_msg)
            
            return normalized_messages
            
        except Exception as e:
            logger.error(f"Error fetching all messages: {e}")
            raise
