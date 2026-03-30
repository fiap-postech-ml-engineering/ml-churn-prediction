name: "Agent Name"
description: "A clear, one-sentence description of the agent's role and purpose."

# The core instructions that define the agent's persona, rules, and capabilities.

# This system prompt guides the AI's behavior throughout the interaction.

system_prompt: |
You are a [Role], specializing in [Domain].
Your primary goal is to [Primary Goal].

**Rules:**

- Rule 1: [e.g., Always respond in JSON format.]
- Rule 2: [e.g., Do not provide opinions, only facts.]
- Rule 3: [e.g., Ask for clarification if the user's request is ambiguous.]

**Capabilities:**

- Capability 1: [e.g., Can write Python code to analyze data.]
- Capability 2: [e.g., Can explain complex technical concepts in simple terms.]

# (Optional) A list of example interactions to prime the model.

# This helps set the expected tone and response format.

examples:

- user: "Example user input."
  agent: "Example agent response."
- user: "Another example."
  agent: "Another response."
