"""
Turn one closed ticket into nodes and edges.

The ticket file only has an id and a text. This file decides the graph
pieces for that ticket:

    nodes: Category, Component, Symptom, RootCause, Solution
    edges: ABOUT, MENTIONS, TOUCHES, HAS_CAUSE, LIVES_IN, SOLVED_BY, POINTS_TO

Two ways to decide those pieces:
    llm: an API key is set. The model reads the ticket and supporting.json
         and returns the nodes and edges as JSON.
    hints: no API key. supporting.json patterns pick the best known story
           by symptom keywords, then this file builds the same JSON shape.

build_graph in graph_db.py does not care which way was used.
It only writes the nodes and edges it is given.

Input: one ticket, supporting.json, and the settings
Output: dict with mode, nodes, and edges
"""

import json

from llm_gen import ask_llm


def extract_ticket(settings, ticket, supporting):
    """
    Choose llm or hints, then return nodes and edges for one ticket.

    Input:
        settings: from config.get_settings()
        ticket: {"id", "text"}
        supporting: the dict from data/supporting.json

    Output:
        {"mode": "llm" or "hints", "nodes": [...], "edges": [...]}
    """
    # Ask the model for nodes and edges, then read that JSON.
    if settings["api_key"]:
        prompt = build_extract_prompt(ticket, supporting)
        answer = ask_llm(settings, prompt)
        graph = parse_extraction(answer)
        return {"mode": "llm", "nodes": graph["nodes"], "edges": graph["edges"]}

    return {"mode": "hints", **extract_with_hints(ticket, supporting)}


def build_extract_prompt(ticket, supporting):
    """
    Tell the model the allowed labels and the ticket it must read.

    supporting.json supplies the allowed names. The patterns in that file
    are not included here, so the model has to read the ticket text.

    Input: one ticket and the supporting dict
    Output: one prompt string
    """
    symptom_names = [item["name"] for item in supporting["symptoms"]]
    category_names = [item["name"] for item in supporting["categories"]]
    return f"""You extract a knowledge graph from one closed support ticket.
Return JSON only. No markdown.

Allowed node labels: {supporting["node_labels"]}
Allowed edge types: {supporting["edge_types"]}
Allowed categories: {category_names}
Allowed components: {supporting["components"]}
Allowed symptom names: {symptom_names}

Use the ticket id {ticket["id"]} as the Ticket node name.
Pick symptom names from the allowed list.
Include every edge that is true for this ticket:
  Ticket ABOUT Category
  Ticket MENTIONS each Symptom
  Ticket TOUCHES Component
  Ticket HAS_CAUSE RootCause
  RootCause LIVES_IN Component
  RootCause SOLVED_BY Solution
  each Symptom POINTS_TO RootCause

Ticket {ticket["id"]}:
{ticket["text"]}

JSON shape:
{{
  "nodes": [
    {{"label": "Category", "name": "Authentication"}},
    {{"label": "Component", "name": "Auth Service"}},
    {{"label": "Symptom", "name": "login_failed"}},
    {{"label": "RootCause", "name": "short cause name", "detail": "one sentence"}},
    {{"label": "Solution", "name": "short fix name", "steps": "one sentence"}}
  ],
  "edges": [
    {{"from_label": "Ticket", "from_name": "{ticket["id"]}", "type": "ABOUT", "to_label": "Category", "to_name": "Authentication"}}
  ]
}}
"""


def parse_extraction(text):
    """
    Read the JSON object out of the model reply.

    Input: raw model text
    Output: dict with "nodes" and "edges"
    The model sometimes wraps the JSON in extra words. The first "{" and
    the last "}" are the object we keep.
    """
    start = text.find("{")
    end = text.rfind("}")
    data = json.loads(text[start:end + 1])
    return {"nodes": data["nodes"], "edges": data["edges"]}


def extract_with_hints(ticket, supporting):
    """
    Build nodes and edges from the best pattern in supporting.json.

    Why this exists:
        The lesson still runs when no API key is set.
        A pattern wins when more of its symptom keywords appear in the text.

    Input: one ticket and the supporting dict
    Output: dict with "nodes" and "edges"
    """
    best = None
    best_score = 0
    for pattern in supporting["patterns"]:
        score = 0
        for symptom_name in pattern["symptoms"]:
            if symptom_keywords_hit(ticket["text"], symptom_name, supporting):
                score += 1
        if score > best_score:
            best = pattern
            best_score = score

    if best is None:
        print("no hint pattern matched")
        return {"nodes": [], "edges": []}
    return pattern_to_graph(ticket, best)


def symptom_keywords_hit(text, symptom_name, supporting):
    """
    Return True when one keyword of this symptom appears in the text.

    Input: ticket text, symptom name, supporting dict
    Output: True or False
    """
    lowered = text.lower()
    for symptom in supporting["symptoms"]:
        if symptom["name"] != symptom_name:
            continue
        for word in symptom["keywords"]:
            if word.lower() in lowered:
                return True
    return False


def pattern_to_graph(ticket, pattern):
    """
    Turn one supporting pattern into the same node and edge lists the
    model is asked to return.

    Input: ticket {"id", "text"} and one pattern dict
    Output: {"nodes": [...], "edges": [...]}
    """
    cause = pattern["root_cause"]
    solution = pattern["solution"]
    nodes = [
        {"label": "Category", "name": pattern["category"]},
        {"label": "Component", "name": pattern["component"]},
        {
            "label": "RootCause",
            "name": cause["name"],
            "detail": cause["detail"],
        },
        {
            "label": "Solution",
            "name": solution["name"],
            "steps": solution["steps"],
        },
    ]
    edges = [
        edge("Ticket", ticket["id"], "ABOUT", "Category", pattern["category"]),
        edge("Ticket", ticket["id"], "TOUCHES", "Component", pattern["component"]),
        edge("Ticket", ticket["id"], "HAS_CAUSE", "RootCause", cause["name"]),
        edge("RootCause", cause["name"], "LIVES_IN", "Component", pattern["component"]),
        edge("RootCause", cause["name"], "SOLVED_BY", "Solution", solution["name"]),
    ]
    for symptom_name in pattern["symptoms"]:
        nodes.append({"label": "Symptom", "name": symptom_name})
        edges.append(edge("Ticket", ticket["id"], "MENTIONS", "Symptom", symptom_name))
        edges.append(edge("Symptom", symptom_name, "POINTS_TO", "RootCause", cause["name"]))
    return {"nodes": nodes, "edges": edges}


def edge(from_label, from_name, rel_type, to_label, to_name):
    """
    Make one edge dict.

    Input: the two ends and the relationship type
    Output: one edge dict in the shape build_graph writes
    """
    return {
        "from_label": from_label,
        "from_name": from_name,
        "type": rel_type,
        "to_label": to_label,
        "to_name": to_name,
    }
