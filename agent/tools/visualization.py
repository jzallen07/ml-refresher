from __future__ import annotations

from cli.api import MLRefresherAPI
from agent.tools import Tool


def make_visualization_tools(api: MLRefresherAPI) -> list[Tool]:
    async def show_visualization(input: dict) -> dict:
        result = api.get_visualization(
            topic=input["topic"],
            name=input["name"],
        )
        if result is None:
            available = api.list_visualizations(topic=input.get("topic"))
            names = [f"{v['topic']}/{v['name']}" for v in available]
            return {"error": f"Visualization not found. Available: {names}"}
        return result

    async def render_heatmap(input: dict) -> dict:
        try:
            path = api.render_heatmap(
                data=input["data"],
                row_labels=input["row_labels"],
                col_labels=input["col_labels"],
                title=input["title"],
                colormap=input.get("colormap", "YlOrRd"),
                annotate=input.get("annotate", True),
            )
            return {"type": "visualization", "image_path": path, "title": input["title"]}
        except Exception as e:
            return {"error": str(e)}

    async def render_function_plot(input: dict) -> dict:
        try:
            path = api.render_function_plot(
                functions=input["functions"],
                x_range=(input.get("x_min", -5.0), input.get("x_max", 5.0)),
                title=input["title"],
                x_label=input.get("x_label", "x"),
                y_label=input.get("y_label", "y"),
            )
            return {"type": "visualization", "image_path": path, "title": input["title"]}
        except Exception as e:
            return {"error": str(e)}

    async def render_tensor_shapes(input: dict) -> dict:
        try:
            path = api.render_tensor_shapes(
                steps=input["steps"],
                title=input["title"],
            )
            return {"type": "visualization", "image_path": path, "title": input["title"]}
        except Exception as e:
            return {"error": str(e)}

    return [
        Tool(
            name="show_visualization",
            description="Display a pre-generated visualization from the library.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic slug (e.g. 'attention_mechanisms', 'loss_functions_and_math')",
                    },
                    "name": {
                        "type": "string",
                        "description": "Visualization name (e.g. 'attention_basic_heatmap')",
                    },
                },
                "required": ["topic", "name"],
            },
            execute=show_visualization,
        ),
        Tool(
            name="render_heatmap",
            description="Render a heatmap visualization (attention weights, confusion matrices, similarity matrices).",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "2D array of numeric values",
                    },
                    "row_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for rows",
                    },
                    "col_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for columns",
                    },
                    "title": {"type": "string", "description": "Chart title"},
                    "colormap": {
                        "type": "string",
                        "description": "Matplotlib colormap name (default: YlOrRd)",
                    },
                    "annotate": {
                        "type": "boolean",
                        "description": "Show numeric values in cells (default: true)",
                    },
                },
                "required": ["data", "row_labels", "col_labels", "title"],
            },
            execute=render_heatmap,
        ),
        Tool(
            name="render_function_plot",
            description="Plot mathematical functions (activation functions, loss curves, learning rate schedules). Expressions use numpy: e.g. 'np.maximum(0, x)', '1 / (1 + np.exp(-x))'.",
            input_schema={
                "type": "object",
                "properties": {
                    "functions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "NumPy expression using variable x, e.g. 'np.maximum(0, x)'",
                                },
                                "label": {"type": "string", "description": "Legend label"},
                            },
                            "required": ["expression", "label"],
                        },
                        "description": "List of functions to plot",
                    },
                    "x_min": {"type": "number", "description": "X-axis minimum (default: -5)"},
                    "x_max": {"type": "number", "description": "X-axis maximum (default: 5)"},
                    "title": {"type": "string", "description": "Chart title"},
                    "x_label": {"type": "string", "description": "X-axis label (default: x)"},
                    "y_label": {"type": "string", "description": "Y-axis label (default: y)"},
                },
                "required": ["functions", "title"],
            },
            execute=render_function_plot,
        ),
        Tool(
            name="render_tensor_shapes",
            description="Visualize tensor dimension flow through neural network layers.",
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Layer/step name"},
                                "shape": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Tensor dimensions, e.g. [32, 128, 512]",
                                },
                            },
                            "required": ["label", "shape"],
                        },
                        "description": "List of steps with tensor shapes",
                    },
                    "title": {"type": "string", "description": "Chart title"},
                },
                "required": ["steps", "title"],
            },
            execute=render_tensor_shapes,
        ),
    ]
