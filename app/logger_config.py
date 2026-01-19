import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Custom Log Level
AGENT_OUTPUT_LEVEL = 25
logging.addLevelName(AGENT_OUTPUT_LEVEL, "AGENT_OUTPUT")

def agent_output(self, message, *args, **kws):
    if self.isEnabledFor(AGENT_OUTPUT_LEVEL):
        self._log(AGENT_OUTPUT_LEVEL, message, args, **kws)

logging.Logger.agent_output = agent_output

def setup_logger():
    """Sets up the comprehensive logging system."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(message)s")

    # 1. All Logs (DEBUG and above)
    all_handler = RotatingFileHandler(log_dir / "all.log", maxBytes=10*1024*1024, backupCount=5)
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)

    # 2. Warnings (WARNING only)
    # We can set level WARNING, which catches WARNING, ERROR, CRITICAL
    warnings_handler = RotatingFileHandler(log_dir / "warnings.log", maxBytes=5*1024*1024, backupCount=3)
    warnings_handler.setLevel(logging.WARNING)
    warnings_handler.setFormatter(detailed_formatter)
    
    # 3. Errors (ERROR and above)
    errors_handler = RotatingFileHandler(log_dir / "errors.log", maxBytes=5*1024*1024, backupCount=3)
    errors_handler.setLevel(logging.ERROR)
    errors_handler.setFormatter(detailed_formatter)

    # 4. Agent Outputs (Only AGENT_OUTPUT level)
    agent_handler = RotatingFileHandler(log_dir / "agent_outputs.log", maxBytes=5*1024*1024, backupCount=5)
    agent_handler.setLevel(AGENT_OUTPUT_LEVEL)
    agent_handler.setFormatter(simple_formatter)
    
    # Filter to ensure ONLY AGENT_OUTPUT level goes here (and maybe higher if we wanted, but request implies separation)
    # This filter ensures we only get our custom level
    class AgentOutputFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == AGENT_OUTPUT_LEVEL

    agent_handler.addFilter(AgentOutputFilter())

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything at root level
    
    # Clear existing handlers to avoid duplicates during reload
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    logger.addHandler(all_handler)
    logger.addHandler(warnings_handler)
    logger.addHandler(errors_handler)
    logger.addHandler(agent_handler)
    
    # Add console handler too for dev visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    logging.info("Logging system initialized.")

def get_logger(name):
    return logging.getLogger(name)
