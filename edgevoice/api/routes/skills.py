import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from edgevoice.core.logging import get_logger
from edgevoice.api.routes import executor

logger = get_logger(__name__)
router = APIRouter()

@router.get("/mcp/tools")
async def get_mcp_tools():
    """
    List all dynamically registered tools from connected MCP servers.
    """
    try:
        tools = await executor.mcp_manager.list_tools()
        return JSONResponse(content={"tools": tools})
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mcp/servers")
async def get_mcp_servers():
    """
    List registered MCP servers and their connection statuses.
    """
    try:
        servers = {}
        config_path = "config/mcp_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                try:
                    config_data = json.load(f)
                    server_configs = config_data.get("mcpServers", {})
                    for name in server_configs:
                        session = executor.mcp_manager._sessions.get(name)
                        servers[name] = "connected" if session else "disconnected"
                except Exception as e:
                    logger.error(f"Failed to read MCP config: {e}")
        return JSONResponse(content={"servers": servers})
    except Exception as e:
        logger.error(f"Failed to get MCP server statuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))
