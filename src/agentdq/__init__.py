"""agentdq: measure how data-quality defects affect ML models and LLM agents, per consumer."""
from agentdq.consumers import Consumer, LLMConsumer, SklearnConsumer
from agentdq.corruption import CANONICAL, ColumnRoles, apply, list_families, register
from agentdq.detectors import detect_all
from agentdq.profile import SensitivityProfile, profile_sensitivity
from agentdq.validate import LeakageError, Report, validate

__all__ = ["CANONICAL", "ColumnRoles", "Consumer", "LLMConsumer", "LeakageError", "Report",
           "SensitivityProfile", "SklearnConsumer", "apply", "detect_all", "list_families",
           "profile_sensitivity", "register", "validate"]
__version__ = "0.1.0a1"
