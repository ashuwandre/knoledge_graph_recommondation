// Paste these into Neo4j Browser at http://localhost:7474
// after you run: python main.py build
// They are the same walks the Python code uses.

// 1) How many nodes of each type?
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS total
ORDER BY label;

// 2) How many arrows of each type?
MATCH ()-[r]->()
RETURN type(r) AS rel, count(r) AS total
ORDER BY rel;

// 3) One path per past ticket: ticket -> cause -> solution.
MATCH (t:Ticket)-[:HAS_CAUSE]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
MATCH (t)-[:ABOUT]->(cat:Category)
RETURN t.id AS ticket_id, cat.name AS category, rc.name AS root_cause, sol.name AS solution
ORDER BY ticket_id;

// 4) Picture this path in the browser.
// Symptom email_not_sent points at the SMTP cause, and that cause has a fix.
MATCH path = (s:Symptom {name: "email_not_sent"})-[:POINTS_TO]->(:RootCause)-[:SOLVED_BY]->(:Solution)
RETURN path;

// 5) Similar tickets for the sample sentence
// "I cannot login and the reset email never comes".
// shared_symptoms is the similarity score.
MATCH (t:Ticket)-[:MENTIONS]->(s:Symptom)
WHERE s.name IN ["login_failed", "email_not_sent"]
MATCH (t)-[:HAS_CAUSE]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
RETURN t.id AS ticket_id,
       rc.name AS root_cause,
       sol.steps AS solution_steps,
       collect(DISTINCT s.name) AS matched_symptoms,
       count(DISTINCT s) AS shared_symptoms
ORDER BY shared_symptoms DESC, ticket_id ASC;
