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

Input of the write functions: tickets (id + text) and the nodes/edges
extracted for each ticket.
Output of the read functions: list of dicts, one dict per result row
"""

from neo4j import GraphDatabase

from extract import extract_ticket


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


def existing_ticket_ids(driver):
    """
    Read the ticket ids already stored in Neo4j.

    Input: open driver
    Output: list of id strings, for example ["T1", "T2"]
    An empty graph returns an empty list.
    """
    rows = run(driver, "MATCH (t:Ticket) RETURN t.id AS id")
    ids = []
    for row in rows:
        ids.append(row["id"])
    return ids


def build_graph(driver, tickets, supporting, settings, incremental=False):
    """
    Write one extracted graph per ticket.

    incremental=False is a full build. The database is cleared first,
    then every ticket is extracted and saved.

    incremental=True keeps the current graph. A ticket id that is already
    stored is skipped. A new id is extracted and merged in. MERGE reuses
    a shared symptom or cause instead of making a second copy.

    Input:
        driver: open connection
        tickets: list of {"id", "text"} from data/tickets.json
        supporting: allowed labels, symptom keywords, and hint patterns
        settings: used to decide llm extraction or hint extraction
        incremental: False wipes the graph, True adds only new ticket ids

    Output:
        Nothing returned. Nodes and arrows now live inside Neo4j.
    """
    # Block: full build starts empty. An add build does not delete anything.
    if not incremental:
        run(driver, "MATCH (n) DETACH DELETE n")
        print("cleared the database")

    # Block: ids already in Neo4j. Only used when we are adding.
    already_there = []
    if incremental:
        already_there = existing_ticket_ids(driver)
        print("already in graph:", already_there)

    allowed_labels = set(supporting["node_labels"])
    allowed_edges = set(supporting["edge_types"])

    for ticket in tickets:
        # A known id is left as it is. Editing an old ticket still needs a full build.
        if ticket["id"] in already_there:
            print("skip", ticket["id"])
            continue

        print("")
        print("EXTRACT", ticket["id"])
        print("INPUT text:", ticket["text"])
        extracted = extract_ticket(settings, ticket, supporting)
        print("MODE:", extracted["mode"])
        save_ticket(driver, ticket)
        for node in extracted["nodes"]:
            save_node(driver, node, allowed_labels)
        for one_edge in extracted["edges"]:
            save_edge(driver, one_edge, allowed_labels, allowed_edges)

    print("")
    print("graph build finished")


def save_ticket(driver, ticket):
    """
    Write the Ticket node from the input file.

    The extractor does not have to return this node. The id and the text
    are already known before extraction.

    Input: driver and one ticket dict
    Output: nothing returned. One Ticket node is stored.
    """
    run(
        driver,
        "MERGE (t:Ticket {id: $id}) SET t.text = $text",
        {"id": ticket["id"], "text": ticket["text"]},
    )
    print(f"  node Ticket({ticket['id']})")


def save_node(driver, node, allowed_labels):
    """
    MERGE one extracted node.

    Input:
        node: {"label", "name", optional "detail", optional "steps"}
        allowed_labels: the label set from supporting.json

    Output: nothing returned.
    Ticket nodes are skipped because save_ticket already wrote them.
    A label that is not in supporting.json is skipped.
    """
    label = node["label"]
    if label == "Ticket" or label not in allowed_labels:
        return

    params = {"name": node["name"]}
    sets = ["n.name = $name"]
    if node.get("detail"):
        params["detail"] = node["detail"]
        sets.append("n.detail = $detail")
    if node.get("steps"):
        params["steps"] = node["steps"]
        sets.append("n.steps = $steps")

    # The label is pasted into the query only after the allowed-label check.
    # Cypher cannot take a label as a $parameter.
    query = f"MERGE (n:{label} {{name: $name}}) SET " + ", ".join(sets)
    run(driver, query, params)
    print(f"  node {label}({node['name']})")


def save_edge(driver, one_edge, allowed_labels, allowed_edges):
    """
    MERGE one extracted arrow between two nodes that were already saved.

    Input:
        one_edge: from_label, from_name, type, to_label, to_name
        allowed_labels and allowed_edges from supporting.json

    Output: nothing returned.
    Ticket ends are matched on id. Every other node is matched on name.
    """
    from_label = one_edge["from_label"]
    to_label = one_edge["to_label"]
    rel_type = one_edge["type"]
    if from_label not in allowed_labels or to_label not in allowed_labels:
        return
    if rel_type not in allowed_edges:
        return

    from_prop = "id" if from_label == "Ticket" else "name"
    to_prop = "id" if to_label == "Ticket" else "name"
    query = f"""
    MATCH (a:{from_label} {{{from_prop}: $from_name}})
    MATCH (b:{to_label} {{{to_prop}: $to_name}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    run(
        driver,
        query,
        {"from_name": one_edge["from_name"], "to_name": one_edge["to_name"]},
    )
    print(
        f"  edge {from_label}({one_edge['from_name']}) "
        f"-[{rel_type}]-> {to_label}({one_edge['to_name']})"
    )


def training_rows(driver):
    """
    Read labeled ticket sentences back out of the graph.

    Why:
        The ticket file has no category field. The category is an ABOUT
        arrow created during extraction. Naive Bayes trains on these rows.

    Input: open driver
    Output: list of {"text", "category"}
    """
    return run(
        driver,
        """
        MATCH (t:Ticket)-[:ABOUT]->(c:Category)
        RETURN t.text AS text, c.name AS category
        ORDER BY t.id
        """,
    )


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
