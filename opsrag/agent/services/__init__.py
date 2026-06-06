"""Agent services -- stateless helpers that support the agent loop.

Currently contains:
    - `toolmsg_compactor` -- summarizes older `tool_result` entries when
      the conversation history grows past a fraction of the model's
      input-token budget, so the loop cap (MAX_TOOL_CALLS) can be raised
      without context blow-up.
"""
