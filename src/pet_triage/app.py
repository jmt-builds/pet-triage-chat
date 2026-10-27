"""The served graph, built once at start-up. Referenced from ``langgraph.json``.

``langgraph dev`` calls an exported function on every run, so the graph is exported as a
built object instead. Tests import ``graph`` rather than this module, so they never need
Ollama or MLflow.

``mlflow.langchain.autolog`` is switched on after the graph is built, so the start-up
warm-up isn't traced. It then traces every run: each node, the prompt the model saw, its raw
reply and the timings. It logs to ``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME``,
which Compose sets.
"""

import mlflow

from pet_triage.graph import make_graph

graph = make_graph()
mlflow.langchain.autolog()
