"""
Step 2. Create the knowledge graph and read it back.

A knowledge graph is a set of nodes and arrows:
    node  = a thing (ticket, symptom, root cause, solution, ...)
    arrow = a fact that ties two things together

Neo4j stores that graph. This file sends Cypher text to Neo4j.
Cypher is Neo4j's query language. Python does not walk the graph itself.

The shape we store:

    (Ticket)-[:ABOUT]->(Category)
    (Ticket)-[:MENTIONS]->(Symptom)
    (Ticket)-[:TOUCHES]->(Component)
    (Ticket)-[:HAS_CAUSE]->(RootCause)
    (RootCause)-[:LIVES_IN]->(Component)
    (RootCause)-[:SOLVED_BY]->(Solution)
    (Symptom)-[:POINTS_TO]->(RootCause)

The last arrow is copied from past tickets. If ticket T3 mentions
email_not_sent and its cause is SMTP credentials expired, we remember
that this symptom can point at that cause.

Input of the write functions: the dataset dict from graph_data.json
Output of the read functions: list of dicts, one dict per result row
"""

from neo4j import GraphDatabase


# Past tickets that share symptom nodes with the new ticket.
# $symptoms is filled by Python. Neo4j does the matching.
# shared_symptoms is the similarity score: more shared symptoms = closer ticket.
SIMILAR_QUERY = """
MATCH (t:Ticket)-[:MENTIONS]->(s:Symptom)
WHERE s.name IN $symptoms
MATCH (t)-[:HAS_CAUSE]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
MATCH (t)-[:ABOUT]->(cat:Category)
MATCH (t)-[:TOUCHES]->(comp:Component)
RETURN t.id AS ticket_id,
       t.text AS ticket_text,
       cat.name AS category,
       comp.name AS component,
       rc.name AS root_cause,
       rc.detail AS cause_detail,
       sol.name AS solution,
       sol.steps AS solution_steps,
       collect(DISTINCT s.name) AS matched_symptoms,
       count(DISTINCT s) AS shared_symptoms
ORDER BY shared_symptoms DESC, ticket_id ASC
LIMIT 3
"""


# Direct walk from the new symptoms to causes, without going through a ticket.
# This is the distilled knowledge. The similar-ticket query is the history.
CAUSE_QUERY = """
MATCH (s:Symptom)-[:POINTS_TO]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
WHERE s.name IN $symptoms
MATCH (rc)-[:LIVES_IN]->(comp:Component)
RETURN s.name AS symptom,
       rc.name AS root_cause,
       rc.detail AS cause_detail,
       sol.name AS solution,
       sol.steps AS solution_steps,
       comp.name AS component
ORDER BY symptom, root_cause
"""


def get_driver(settings):
    """
    Open a connection to Neo4j.

    Input:
        settings dict with neo4j_uri, neo4j_user, neo4j_password

    Output:
        a driver object. Pass it to the other functions in this file.
        Call driver.close() when the script is finished.
    """
    return GraphDatabase.driver(
        settings["neo4j_uri"],
        auth=(settings["neo4j_user"], settings["neo4j_password"]),
    )


def run(driver, cypher, params=None):
    """
    Send one Cypher query and collect every row.

    Input:
        driver: open connection
        cypher: query text
        params: dict of values referenced as $name inside the query

    Output:
        list of dicts. Each dict is one result row.
        An empty list means the pattern was not found.

    The result is read inside the session on purpose. Neo4j streams rows,
    and the stream is closed when the session closes.
    """
    if params is None:
        params = {}
    with driver.session() as session:
        result = session.run(cypher, params)
        return [row.data() for row in result]


def build_graph(driver, data):
    """
    Wipe the database and create the whole graph from the dataset.

    Why wipe:
        This is a learning sandbox. Re-running build should show the same
        graph, not a second copy of every node. MERGE also protects us,
        and the wipe makes deleted dataset rows disappear too.

    Input:
        driver: open connection
        data: the dict loaded from data/graph_data.json

    Output:
        Nothing returned. Nodes and arrows now live inside Neo4j.
        Progress lines are printed so you can see each ticket being linked.

    MERGE means "make this node if it is missing, otherwise reuse it".
    The property inside the braces is the identity. Same name = same node.
    A symptom used by three tickets is still one Symptom node.
    """
    # Block: start from an empty database.
    run(driver, "MATCH (n) DETACH DELETE n")
    print("cleared the database")

    # Block: create the plain lookup nodes. No arrows yet.
    for category in data["categories"]:
        run(driver, "MERGE (:Category {name: $name})", {"name": category["name"]})
    for name in data["components"]:
        run(driver, "MERGE (:Component {name: $name})", {"name": name})
    for symptom in data["symptoms"]:
        run(driver, "MERGE (:Symptom {name: $name})", {"name": symptom["name"]})
    for solution in data["solutions"]:
        run(
            driver,
            "MERGE (s:Solution {name: $name}) SET s.steps = $steps",
            {"name": solution["name"], "steps": solution["steps"]},
        )
    print(
        "lookup nodes:",
        len(data["categories"]), "categories,",
        len(data["components"]), "components,",
        len(data["symptoms"]), "symptoms,",
        len(data["solutions"]), "solutions",
    )

    # Block: create each cause and attach the component and the solution.
    # The cause name is the identity. detail/component/solution are facts.
    for cause in data["root_causes"]:
        run(
            driver,
            """
            MERGE (rc:RootCause {name: $name})
            SET rc.detail = $detail
            WITH rc
            MATCH (comp:Component {name: $component})
            MATCH (sol:Solution {name: $solution})
            MERGE (rc)-[:LIVES_IN]->(comp)
            MERGE (rc)-[:SOLVED_BY]->(sol)
            """,
            {
                "name": cause["name"],
                "detail": cause["detail"],
                "component": cause["component"],
                "solution": cause["solution"],
            },
        )
    print("root causes:", len(data["root_causes"]))

    # Block: create each past ticket and its arrows.
    # POINTS_TO is derived here from history: a symptom that showed up
    # on a ticket can point at that ticket's cause.
    for ticket in data["tickets"]:
        run(
            driver,
            """
            MERGE (t:Ticket {id: $id})
            SET t.text = $text
            WITH t
            MATCH (cat:Category {name: $category})
            MATCH (comp:Component {name: $component})
            MATCH (rc:RootCause {name: $root_cause})
            MERGE (t)-[:ABOUT]->(cat)
            MERGE (t)-[:TOUCHES]->(comp)
            MERGE (t)-[:HAS_CAUSE]->(rc)
            """,
            {
                "id": ticket["id"],
                "text": ticket["text"],
                "category": ticket["category"],
                "component": ticket["component"],
                "root_cause": ticket["root_cause"],
            },
        )
        for symptom_name in ticket["symptoms"]:
            run(
                driver,
                """
                MATCH (t:Ticket {id: $id})
                MATCH (s:Symptom {name: $symptom})
                MATCH (rc:RootCause {name: $root_cause})
                MERGE (t)-[:MENTIONS]->(s)
                MERGE (s)-[:POINTS_TO]->(rc)
                """,
                {
                    "id": ticket["id"],
                    "symptom": symptom_name,
                    "root_cause": ticket["root_cause"],
                },
            )
        print(f"  {ticket['id']} --HAS_CAUSE--> {ticket['root_cause']}")

    print("graph build finished")


def _sort_symptom_lists(rows):
    """
    Sort the symptom names inside each result row.

    Why:
        Neo4j's collect() does not promise an order. Sorting makes the
        printout and the model prompt stable every time you run the script.

    Input: rows from SIMILAR_QUERY
    Output: the same list, with matched_symptoms sorted in place
    """
    for row in rows:
        row["matched_symptoms"] = sorted(row["matched_symptoms"])
    return rows


def similar_tickets(driver, symptom_names):
    """
    Find up to 3 past tickets that share symptoms with the new ticket.

    This is the graph similarity. We do not compare sentences with math.
    Two tickets are similar when they both have a MENTIONS arrow to the
    same Symptom node. The score is how many of those nodes they share.

    Input:
        symptom_names: list of strings from the classifier, e.g.
        ["login_failed", "email_not_sent"]

    Output:
        list of dicts ordered by shared_symptoms, highest first.
        Each dict includes the past ticket text, its cause, and its solution.
    """
    rows = run(driver, SIMILAR_QUERY, {"symptoms": symptom_names})
    return _sort_symptom_lists(rows)


def causes_for_symptoms(driver, symptom_names):
    """
    Walk Symptom -POINTS_TO-> RootCause -SOLVED_BY-> Solution.

    Why a second query:
        similar_tickets answers "what happened before?"
        this query answers "what do we believe this symptom can mean?"
        One symptom can point at several causes. The similar-ticket score
        is what tells those causes apart.

    Input: the same symptom name list
    Output: list of dicts, one per symptom-to-cause arrow
    """
    return run(driver, CAUSE_QUERY, {"symptoms": symptom_names})


def show_graph(driver, full=False):
    """
    Print what is stored, so you can see the graph without the browser.

    Input:
        full=False prints counts and the 10 ticket paths.
        full=True also prints every arrow.

    Output:
        Nothing returned. Text goes to the terminal.
    """
    # Block: how many nodes of each type.
    print("")
    print("NODE COUNTS")
    counts = run(
        driver,
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS total
        ORDER BY label
        """,
    )
    for row in counts:
        print(f"  {row['label']}: {row['total']}")

    # Block: how many arrows of each type.
    print("")
    print("RELATIONSHIP COUNTS")
    rels = run(
        driver,
        """
        MATCH ()-[r]->()
        RETURN type(r) AS rel, count(r) AS total
        ORDER BY rel
        """,
    )
    for row in rels:
        print(f"  {row['rel']}: {row['total']}")

    # Block: one readable path per past ticket.
    # This is the walk the analyzer uses: ticket -> cause -> solution.
    print("")
    print("TICKET PATHS")
    paths = run(
        driver,
        """
        MATCH (t:Ticket)-[:HAS_CAUSE]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
        MATCH (t)-[:ABOUT]->(cat:Category)
        RETURN t.id AS ticket_id, cat.name AS category, rc.name AS root_cause, sol.name AS solution
        ORDER BY ticket_id
        """,
    )
    for row in paths:
        print(
            f"  {row['ticket_id']} | {row['category']} | "
            f"{row['root_cause']} | fix: {row['solution']}"
        )

    if not full:
        return

    # Block: every arrow, for when you want to read the whole graph as text.
    print("")
    print("ALL RELATIONSHIPS")
    arrows = run(
        driver,
        """
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] AS a_label,
               coalesce(a.id, a.name) AS a_name,
               type(r) AS rel,
               labels(b)[0] AS b_label,
               coalesce(b.id, b.name) AS b_name
        ORDER BY a_label, a_name, rel, b_name
        """,
    )
    for row in arrows:
        print(
            f"  {row['a_label']}({row['a_name']}) "
            f"-[{row['rel']}]-> "
            f"{row['b_label']}({row['b_name']})"
        )
