"""Semiconductor Agent: tool-use tasks in executable semiconductor environments.

197 held-out tasks, each bound to its own environment (record database,
workspace files, task-specific tools backed by a pinned Python software
profile). Generation and grading are separate stages:

1. ``python -m micius_eval.semiconductor_agent.run``     -- the model solves tasks
2. ``python -m micius_eval.semiconductor_agent.verify``  -- frozen-rubric grading
"""
