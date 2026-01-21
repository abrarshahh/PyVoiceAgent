from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.nodes import transcribe_audio, process_input, synthesize_audio, save_conversation

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("process", process_input)
workflow.add_node("synthesize", synthesize_audio)
workflow.add_node("save_conversation", save_conversation)

# Define edges
# Start at transcribe. If no audio, it passes through to process.
workflow.set_entry_point("transcribe")

workflow.add_edge("transcribe", "process")
workflow.add_edge("process", "synthesize")
workflow.add_edge("synthesize", "save_conversation")
workflow.add_edge("save_conversation", END)

# Compile the graph
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
