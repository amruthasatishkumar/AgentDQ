"""agentdq: measure how data-quality defects affect ML models and LLM agents, per consumer."""
from agentdq.consumers import Consumer, LLMConsumer, SklearnConsumer
from agentdq.corruption import CANONICAL, ColumnRoles, apply, list_families, register

__all__ = ["CANONICAL", "ColumnRoles", "Consumer", "LLMConsumer", "SklearnConsumer",
           "apply", "list_families", "register"]
__version__ = "0.0.1"
