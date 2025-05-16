format_ops = dict(
            preprompt = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            query_bag = 'I',
            reply_beg = '\n+=*/=O',
            lines_sep = '\n',
            max_tokens = 128000,
        )

system_prompt = ''
'''You are an ARC-AGI specialist with doctoral-level abstract reasoning skills. Your problem-solving process MUST follow these steps:
1. Pattern Identification: Analyze color distribution, spatial relationships and symmetries, and size relationships between input and output
2. Transformation Taxonomy: Classify changes as either 
   - Object Manipulation (rotation/scaling/reflection)
   - Color Mapping (palette reassignment)
   - Structural Recomposition (grid expansion/contraction)
3. Hypothesis Generation: Propose 3 distinct transformation rules
4. Cross-Validation: Verify each hypothesis against ALL training examples
5. Final selection: Decide on the most reliable hypothesis as the final output'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template1 = ''
'''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------'''
user_message_template2 = ''
'''----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
----------------------------------------'''

user_message_template3 = ''
'''----------------------------------------
What is the output grid? Provide only the following two items. Do not provide additional information.
- The size of the output in the form of "(width, height)"
- The output grid in the form as the example input and output pair'''

# Below functions are used to visualize the grid data in a more human-readable format
# imported from skeleton/utils.py

from pathlib import Path

import numpy as np

from rich.console import Console
from rich.text import Text
from typing import List

color_map = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
    8: "bright_red",
    9: "bright_green",
}

console = Console()

def make_rich_lines(grid: List[List[int]]) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append("  ", style=f"on {color}")
        raw = Text("  " + str(row))
        visual.append(raw)
        lines.append(visual)
    return lines

def render_grid(grid: List[List[int]]):
    lines = make_rich_lines(grid)
    for line in lines:
        console.print(line)
