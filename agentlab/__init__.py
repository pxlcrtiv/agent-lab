"""agent-lab: a lean, dependency-free AI agent framework in pure Python.

Demonstrates the patterns used in production agent stacks:
tool use / function calling, memory and context handling, and
workflow automation -- with no third-party runtime dependencies.
"""

__version__ = "0.1.0"

from .agent import Agent, AgentResult  # noqa: F401
from .llm import MockBackend, OpenAICompatBackend  # noqa: F401
from .memory import Memory  # noqa: F401
from .tools import Tool, ToolRegistry, tool  # noqa: F401
from .workflow import Workflow, WorkflowResult  # noqa: F401