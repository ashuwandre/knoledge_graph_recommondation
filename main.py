"""
Ticket analysis learning project.

What this script does, in order:
    1. Classify a new support ticket into one class (multiclass).
    2. Search the Neo4j knowledge graph for similar past tickets
       and for symptom -> cause -> solution paths.
    3. Generate a root cause and a solution.
       With an API key, a language model writes it from the graph context.
       Without a key, the top graph match is copied so you can still learn.

Commands:
    python main.py build
    python main.py show
    python main.py analyze
    python main.py analyze "customer was charged two times"
    python main.py demo

demo = build, then a short show, then analyze the 3 sample tickets.
"""

import json
import sys

from classify import classify_ticket, train_category_model
from config import DATA_FILE, NEW_FILE, get_settings
from graph_db import (
    CAUSE_QUERY,
    SIMILAR_QUERY,
    build_graph,
    causes_for_symptoms,
    get_driver,
    show_graph,
    similar_tickets,
)
from llm_gen import generate


HELP = """
Commands:
  python main.py build
  python main.py show
  python main.py analyze
  python main.py analyze "your ticket text here"
  python main.py demo

Start Neo4j first:  docker compose up -d
Then open http://localhost:7474  (user neo4j, password ticket123)
"""


def load_json(path):
    """
    Read one JSON file.

    Input: path to a .json file
    Output: the Python object inside it (dict or list)
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def print_step(title):
    """
    Print a visible step header so the terminal matches the README.

    Input: short title string
    Output: nothing returned
    """
    print("")
    print("-" * 60)
    print(title)
    print("-" * 60)


def analyze_one(driver, data, ticket, settings, model):
    """
    Run the three steps for a single new ticket and print every input
    and output.

    Input:
        driver: open Neo4j connection
        data: full dataset (keyword lists live here)
        ticket: dict with at least id and text
        settings: connection and model settings
        model: category model already trained on the past tickets

    Output:
        Nothing returned. The analysis is printed.
        expected_category / expected_cause are printed at the end as a
        study check. They are not passed to the classifier or the model.
    """
    text = ticket["text"]
    print("")
    print("=" * 60)
    print("NEW TICKET", ticket["id"])
    print("=" * 60)
    print("INPUT text:", text)

    # ----- Step 1. Multiclass classification with Naive Bayes. -----
    print_step("STEP 1  classify the ticket (multiclass ML)")
    print("INPUT: ticket text + Naive Bayes model trained on the past tickets")
    classification = classify_ticket(text, model, data["symptoms"])
    print("OUTPUT category probabilities:")
    for row in classification["category_scores"]:
        print(f"  {row['name']} {row['score']:.2f}")
    print("OUTPUT winner:", classification["winner"]["name"])
    print("OUTPUT symptoms:")
    if not classification["symptoms"]:
        print("  (none)")
    for symptom in classification["symptoms"]:
        print(f"  {symptom['name']} hits {symptom['hits']}")

    symptom_names = [item["name"] for item in classification["symptoms"]]

    # ----- Step 2. Walk the graph. History query, then knowledge query. -----
    print_step("STEP 2  search the knowledge graph")
    print("INPUT symptoms:", symptom_names)
    print("INPUT cypher (similar past tickets):")
    print(SIMILAR_QUERY.strip())
    similar_rows = similar_tickets(driver, symptom_names)
    print("OUTPUT similar tickets:")
    if not similar_rows:
        print("  (none)")
    for row in similar_rows:
        print(
            f"  {row['ticket_id']} shared {row['shared_symptoms']} "
            f"symptoms {row['matched_symptoms']}"
        )
        print(f"    text: {row['ticket_text']}")
        print(f"    category: {row['category']} | component: {row['component']}")
        print(f"    cause: {row['root_cause']}")
        print(f"    solution: {row['solution_steps']}")

    print("INPUT cypher (symptom to cause):")
    print(CAUSE_QUERY.strip())
    cause_rows = causes_for_symptoms(driver, symptom_names)
    print("OUTPUT cause links:")
    if not cause_rows:
        print("  (none)")
    for row in cause_rows:
        print(
            f"  {row['symptom']} -> {row['root_cause']} "
            f"| {row['component']} | {row['solution']}"
        )

    # ----- Step 3. Write the answer from the graph context. -----
    print_step("STEP 3  generate root cause and solution")
    result = generate(settings, text, classification, similar_rows, cause_rows)
    print("MODE:", result["mode"])
    print("INPUT prompt:")
    print(result["prompt"].rstrip())
    if result["raw"]:
        print("OUTPUT raw model text:")
        print(result["raw"].rstrip())
    print("OUTPUT answer:")
    for key in ("CATEGORY", "ROOT CAUSE", "SOLUTION", "WHY"):
        print(f"  {key}: {result['parsed'][key]}")

    # Study check only. These fields never enter step 1, 2, or 3.
    if ticket.get("expected_category"):
        print("STUDY NOTE (not used by the code):")
        print("  expected category:", ticket["expected_category"])
        print("  expected cause:", ticket["expected_cause"])


def main():
    """
    Read the command and run build, show, analyze, or demo.

    Input: sys.argv
    Output: nothing returned. Database changes only happen on build and demo.
    """
    if len(sys.argv) < 2:
        print(HELP)
        return

    command = sys.argv[1]
    settings = get_settings()
    driver = get_driver(settings)

    # One ping. If Neo4j is down, this line raises and the script stops.
    driver.verify_connectivity()
    print("connected to", settings["neo4j_uri"])

    data = load_json(DATA_FILE)

    # The category model reads the past tickets only. Build and show do not need it.
    model = None
    if command in ("analyze", "demo"):
        model = train_category_model(data["tickets"])
        print("trained Naive Bayes on", len(data["tickets"]), "past tickets")
        print("classes:", ", ".join(model.classes_))

    if command == "build":
        build_graph(driver, data)
    elif command == "show":
        show_graph(driver, full=True)
    elif command == "analyze":
        if len(sys.argv) > 2:
            custom = {
                "id": "CUSTOM",
                "text": " ".join(sys.argv[2:]),
            }
            analyze_one(driver, data, custom, settings, model)
        else:
            for ticket in load_json(NEW_FILE):
                analyze_one(driver, data, ticket, settings, model)
    elif command == "demo":
        build_graph(driver, data)
        show_graph(driver, full=False)
        for ticket in load_json(NEW_FILE):
            analyze_one(driver, data, ticket, settings, model)
    else:
        print(HELP)

    driver.close()


if __name__ == "__main__":
    main()
