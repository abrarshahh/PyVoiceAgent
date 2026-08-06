import json
import os
import asyncio
from typing import Dict, List, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from edgevoice.core.logging import get_logger

logger = get_logger(__name__)

class MCPManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, config_path: str = "config/mcp_config.json"):
        self.config_path = config_path
        self._sessions: Dict[str, ClientSession] = {}
        self._contexts = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        async with self._lock:
            if self._initialized:
                return
            
            if not os.path.exists(self.config_path):
                logger.warning(f"MCP config not found at {self.config_path}")
                return
                
            with open(self.config_path, "r") as f:
                try:
                    config = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to parse MCP config: {e}")
                    return
                
            servers = config.get("mcpServers", {})
            for name, srv_config in servers.items():
                try:
                    command = srv_config.get("command")
                    args = srv_config.get("args", [])
                    env = srv_config.get("env")
                    
                    # Resolve environment variables
                    full_env = {**os.environ}
                    if env:
                        for k, v in env.items():
                            full_env[k] = os.path.expandvars(str(v))
                    
                    logger.info(f"Starting MCP server: {name} ({command} {' '.join(args)})")
                    
                    params = StdioServerParameters(
                        command=command,
                        args=args,
                        env=full_env
                    )
                    
                    ctx = stdio_client(params)
                    read_stream, write_stream = await ctx.__aenter__()
                    
                    session = ClientSession(read_stream, write_stream)
                    await session.__aenter__()
                    await session.initialize()
                    
                    self._contexts[name] = ctx
                    self._sessions[name] = session
                    logger.info(f"MCP server '{name}' initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to start MCP server '{name}': {e}")
                    
            self._initialized = True

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Retrieve all tools from all active MCP servers."""
        await self.initialize()
        all_tools = []
        for server_name, session in self._sessions.items():
            try:
                res = await session.list_tools()
                for t in res.tools:
                    all_tools.append({
                        "server_name": server_name,
                        "name": t.name,
                        "description": t.description,
                        "input_schema": getattr(t, "input_schema", getattr(t, "inputSchema", {}))
                    })
            except Exception as e:
                logger.error(f"Failed to list tools for '{server_name}': {e}")
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool on a specific MCP server."""
        await self.initialize()
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"MCP server '{server_name}' is not connected.")
        
        try:
            res = await session.call_tool(tool_name, arguments)
            # Combine text fields from the content list
            output_texts = []
            for content in res.content:
                # content can be a TextContent object from mcp types
                if hasattr(content, "text"):
                    output_texts.append(content.text)
                elif isinstance(content, dict) and "text" in content:
                    output_texts.append(content["text"])
            return "\n".join(output_texts)
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}' on '{server_name}': {e}")
            raise e

    async def shutdown(self):
        async with self._lock:
            for name, session in list(self._sessions.items()):
                try:
                    await session.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error shutting down session for '{name}': {e}")
            for name, ctx in list(self._contexts.items()):
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error shutting down transport context for '{name}': {e}")
            self._sessions.clear()
            self._contexts.clear()
            self._initialized = False
            logger.info("MCP Manager shutdown complete.")
