from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.workflows.state import AgentState

# Import nodes from their dedicated locations
from app.agents.assistant import process_input
from app.tools.transcriber import transcribe_audio
from app.tools.segmenter import segment_text
from app.tools.refiner import refine_and_guardrail
from app.tools.synthesizer import synthesize_audio
from app.tools.archiver import save_conversation

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("process", process_input)
workflow.add_node("segment", segment_text)
workflow.add_node("refine", refine_and_guardrail)
workflow.add_node("synthesize", synthesize_audio)
workflow.add_node("save_conversation", save_conversation)

# Define edges
# Start at transcribe. If no audio, it passes through to process.
workflow.set_entry_point("transcribe")

workflow.add_edge("transcribe", "process")
workflow.add_edge("process", "segment")
workflow.add_edge("segment", "refine")
workflow.add_edge("refine", "synthesize")
workflow.add_edge("synthesize", "save_conversation")
workflow.add_edge("save_conversation", END)

# Compile the graph
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

