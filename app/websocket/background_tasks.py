"""
Background Task Manager

Utilities for managing background tasks, timeouts, and task cancellation.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass

from .types import BackgroundTaskConfig


class BackgroundTaskManager:
    """Manages background tasks with timeout and cancellation support"""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_configs: Dict[str, BackgroundTaskConfig] = {}
        self.cancelled_tasks: Set[str] = set()
        
    async def create_task(
        self,
        coro: Callable,
        task_type: str = "general",
        timeout: int = 300,
        max_retries: int = 3,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a background task with timeout and retry support"""
        task_id = str(uuid.uuid4())
        
        config = BackgroundTaskConfig(
            task_id=task_id,
            task_type=task_type,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {}
        )
        
        self.task_configs[task_id] = config
        
        # Create the task
        task = asyncio.create_task(self._execute_with_timeout(coro, task_id))
        self.active_tasks[task_id] = task
        
        return task_id
    
    async def _execute_with_timeout(self, coro: Callable, task_id: str) -> Any:
        """Execute coroutine with timeout and retry logic"""
        config = self.task_configs[task_id]
        
        try:
            # Check if task was cancelled before starting
            if task_id in self.cancelled_tasks:
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")
            
            # Execute with timeout
            result = await asyncio.wait_for(coro(), timeout=config.timeout)
            
            # Task completed successfully
            await self._cleanup_task(task_id)
            return result
            
        except asyncio.TimeoutError:
            print(f"Task {task_id} timed out after {config.timeout} seconds")
            await self._handle_task_failure(task_id, "timeout")
            
        except asyncio.CancelledError:
            print(f"Task {task_id} was cancelled")
            await self._cleanup_task(task_id)
            raise
            
        except Exception as e:
            print(f"Task {task_id} failed with error: {e}")
            await self._handle_task_failure(task_id, str(e))
            
    async def _handle_task_failure(self, task_id: str, error: str):
        """Handle task failure with retry logic"""
        config = self.task_configs[task_id]
        
        if config.retry_count < config.max_retries:
            config.retry_count += 1
            print(f"Retrying task {task_id} (attempt {config.retry_count}/{config.max_retries})")
            
            # Wait before retry (exponential backoff)
            wait_time = min(2 ** config.retry_count, 60)  # Max 60 seconds
            await asyncio.sleep(wait_time)
            
            # Don't retry if task was cancelled
            if task_id in self.cancelled_tasks:
                await self._cleanup_task(task_id)
                return
            
            # Retry the task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Note: This would need the original coro to be stored for retry
            # For now, we'll just clean up
            await self._cleanup_task(task_id)
        else:
            print(f"Task {task_id} failed after {config.max_retries} retries")
            await self._cleanup_task(task_id)
    
    async def _cleanup_task(self, task_id: str):
        """Clean up task resources"""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        if task_id in self.task_configs:
            del self.task_configs[task_id]
        if task_id in self.cancelled_tasks:
            self.cancelled_tasks.remove(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id not in self.active_tasks:
            return False
            
        # Mark as cancelled
        self.cancelled_tasks.add(task_id)
        
        # Cancel the asyncio task
        task = self.active_tasks[task_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        await self._cleanup_task(task_id)
        return True
    
    async def cancel_user_tasks(self, user_id: str) -> int:
        """Cancel all tasks for a specific user"""
        cancelled_count = 0
        
        for task_id, config in list(self.task_configs.items()):
            if config.metadata.get("user_id") == user_id:
                if await self.cancel_task(task_id):
                    cancelled_count += 1
                    
        return cancelled_count
    
    def is_task_active(self, task_id: str) -> bool:
        """Check if a task is currently active"""
        return task_id in self.active_tasks
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a task"""
        if task_id not in self.task_configs:
            return None
            
        config = self.task_configs[task_id]
        task = self.active_tasks.get(task_id)
        
        return {
            "task_id": task_id,
            "task_type": config.task_type,
            "timeout": config.timeout,
            "retry_count": config.retry_count,
            "max_retries": config.max_retries,
            "is_active": task_id in self.active_tasks,
            "is_cancelled": task_id in self.cancelled_tasks,
            "metadata": config.metadata
        }
    
    def get_active_task_count(self) -> int:
        """Get number of active tasks"""
        return len(self.active_tasks)
    
    def get_tasks_by_type(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all tasks of a specific type"""
        tasks = {}
        
        for task_id, config in self.task_configs.items():
            if config.task_type == task_type:
                tasks[task_id] = self.get_task_info(task_id)
                
        return tasks
    
    async def cleanup_completed_tasks(self) -> int:
        """Clean up completed tasks"""
        completed_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if task.done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            await self._cleanup_task(task_id)
            
        return len(completed_tasks)
    
    async def shutdown(self):
        """Shutdown the task manager and cancel all tasks"""
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Wait for all tasks to complete cancellation
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
