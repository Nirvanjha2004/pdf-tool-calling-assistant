"""
Calculator Tool — "Hello World" of Function Calling

A safe math expression evaluator. The LLM calls this tool
whenever it needs to perform calculations.

Uses Python's AST module to safely evaluate math expressions
without using dangerous eval().
"""

import ast
import math
import operator
from typing import Any


# ─── Safe Expression Evaluator ───────────────────────────────────────────────

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node safely."""

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")

    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.Attribute):
        # Handle math.pi, math.e, etc.
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            attr_name = node.attr
            if hasattr(math, attr_name) and not callable(getattr(math, attr_name)):
                return getattr(math, attr_name)
            raise ValueError(f"Unknown math constant: math.{attr_name}")
        raise ValueError(f"Unsupported attribute access: {type(node).__name__}")

    if isinstance(node, ast.Call):
        # Allow math.sqrt(), math.sin(), math.cos(), etc.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            func_name = node.func.attr
            if not hasattr(math, func_name):
                raise ValueError(f"Unknown math function: math.{func_name}")
            args = [_eval_node(arg) for arg in node.args]
            return getattr(math, func_name)(*args)

        raise ValueError("Only math.* functions are allowed")

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, %, //, math.sqrt(), math.sin(),
              math.cos(), math.tan(), math.log(), math.pi, math.e, etc.

    Args:
        expression: Math expression string, e.g. "2 + 2", "math.sqrt(144)"

    Returns:
        Result as string, or error message if evaluation fails.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        # Format nicely — avoid trailing zeros for floats
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ─── Tool Definition (JSON Schema for the LLM) ───────────────────────────────

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for arithmetic, trigonometry, or any math calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate. "
                        "Examples: '2 + 2', 'math.sqrt(144)', '(15 * 3) / 5', "
                        "'math.sin(math.pi / 2)', '2 ** 10'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}


def handle_tool_call(arguments: dict) -> str:
    """Execute the calculator tool with the given arguments."""
    expression = arguments.get("expression", "")
    return calculate(expression)
