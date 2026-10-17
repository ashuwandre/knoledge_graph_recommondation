# Ticket analysis with a knowledge graph

A small learning project. It classifies a support ticket, finds similar past tickets in Neo4j, and writes a root cause and a solution.

There are 10 past tickets and 3 new ones. The code is plain Python: functions, Cypher strings, and print statements. There is no error handling, no web app, and no machine-learning library.

## Run it

You need Python 3.10+ and Docker Desktop.

```powershell
docker compose up -d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py demo
```

If PowerShell blocks the activate script, run the file directly:

```powershell
.\.venv\Scripts\python.exe main.py demo
```

Wait until http://localhost:7474 opens before `demo`. First start can take a minute.

Login in the browser: user `neo4j`, password `ticket123`, URI `bolt://localhost:7687`.

Other commands:

```powershell
python main.py build
python main.py add
python main.py show
python main.py analyze
python main.py analyze "I cannot login and the reset email never comes"
```

`build` deletes the graph and loads every ticket again. `add` keeps the current graph and writes only ticket ids that are not already stored. `show` prints the graph. `analyze` runs the 3 new tickets, or one sentence you type. `demo` does build, a short show, then analyze.

The graph steps work with no API key. The script then copies the closest past ticket. To let a model write the wording, copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Any OpenAI-compatible server works. The Ollama example is in `.env.example`.

## What you will learn

- What a node, a label, a property, and a relationship are.
- How a JSON dataset becomes a graph with `MERGE`.
- How a small Naive Bayes model picks the ticket class from past tickets.
- How a new ticket is turned into symptom names.
- How similarity is "how many symptom nodes do we share".
- How those rows are pasted into a prompt so the model writes a root cause and a solution from the graph, not from memory.

## Thought process

Support tickets repeat. "Cannot login, reset email never arrives" has shown up before, and last time the cause was an expired mail password.

A spreadsheet can store one row per ticket. The useful part of this problem is the links: this sentence mentions these symptoms, those symptoms showed up on an old ticket, that ticket had a cause, and that cause has a fix. A knowledge graph stores those links as arrows, so the program can walk them.

The walk we wanted is:

```
new sentence
  -> symptom names
  -> past tickets that mention those symptoms
  -> their root cause
  -> the solution attached to that cause
```

A language model is good at writing a clear answer and weak at remembering your 10 old tickets. So the model is last. The graph finds the facts. The model only writes them up. If two causes are possible, the prompt tells the model to prefer the past ticket that shares more symptoms.

Classification comes first because the graph does not store raw sentences on the query side. It stores names like `email_not_sent`. Step 1 turns the new sentence into those names. The class (Authentication, Payment, Performance, Data) is the multiclass label.

## What was created

| File | Job |
| --- | --- |
| `data/tickets.json` | The only graph input: 10 closed tickets, each with an id and a text |
| `data/supporting.json` | Allowed labels, symptom keywords, and hint patterns used while the graph is created |
| `extract.py` | Reads one ticket and returns nodes and edges |
| `data/new_tickets.json` | 3 unlabeled tickets to practice on, plus a study answer that the code does not use |
| `docker-compose.yml` | Starts Neo4j |
| `config.py` | Reads the database address and the optional model key |
| `classify.py` | Step 1. Naive Bayes class probabilities, plus symptom names |
| `graph_db.py` | Writes the extracted nodes and edges into Neo4j, then searches them |
| `llm_gen.py` | Step 3. Builds the prompt and writes the root cause and solution |
| `main.py` | Connects the steps and prints every input and output |
| `cypher/examples.cypher` | The same queries, ready to paste into Neo4j Browser |
| `.env.example` | Where a model key goes when you want one |
| `requirements.txt` | `neo4j`, `openai`, and `scikit-learn` |

Call order inside `python main.py demo`:

```
main
  build_graph
    extract_ticket     (LLM, or hint patterns when no API key)
    save each node and edge
  show_graph
  train_from_graph
  analyze_one          (once per new ticket)
    classify_ticket
    similar_tickets
    causes_for_symptoms
    generate
      build_prompt
      answer_from_graph    when no API key
      ask_llm              when an API key is set
```

## Words used in Neo4j

| Word | In this project |
| --- | --- |
| Node | One thing. Example: the symptom `email_not_sent` |
| Label | The type of a node. Example: `:Symptom`, `:Ticket`, `:RootCause` |
| Property | A field on a node. Example: `name`, `text`, `steps` |
| Relationship | The arrow. Example: `-[:MENTIONS]->` |
| Cypher | The query language sent to Neo4j |
| `MERGE` | Create the node if it is missing, otherwise reuse the one with the same name |
| `MATCH` | Find nodes that fit a pattern |
| Traversal | Following arrows from one node to the next |

Python does not contain the graph. Python sends Cypher, and Neo4j stores and walks the graph.

## The graph

```
(Ticket)-[:ABOUT]->(Category)
(Ticket)-[:MENTIONS]->(Symptom)
(Ticket)-[:TOUCHES]->(Component)
(Ticket)-[:HAS_CAUSE]->(RootCause)
(RootCause)-[:LIVES_IN]->(Component)
(RootCause)-[:SOLVED_BY]->(Solution)
(Symptom)-[:POINTS_TO]->(RootCause)
```

`POINTS_TO` is not typed by hand. While each past ticket is loaded, every symptom on that ticket gets an arrow to that ticket's cause. Three different login tickets therefore make `login_failed` point at three different causes. That is why one symptom is not enough, and the shared-symptom count is what picks a winner.

The login corner of the graph looks like this:

```
                 (login_failed)
                /      |        \
               v       v         v
           (hash)    (ttl)      (smtp) <---- (email_not_sent)
             |         |          |
          (clear     (longer    (rotate
           cache)     token)     mail password)

T1 mentions login_failed + password_reset  -> hash
T3 mentions login_failed + email_not_sent  -> smtp
```

A new ticket with both `login_failed` and `email_not_sent` shares 2 symptoms with T3 and 1 symptom with T1. T3 wins. That is the whole similarity idea.

## The dataset

`data/tickets.json` is the only ticket input. Each item has an `id` and a `text`. The text is a closed ticket: the complaint, what was found, and the fix. It does not contain a category field or a list of edges.

`data/supporting.json` is used while the graph is created:

- `node_labels` and `edge_types` are the only kinds of nodes and arrows the builder will save.
- `categories`, `components`, and `symptoms` are the allowed names. Symptom `keywords` also find those names in a new ticket during analysis.
- `patterns` are used only when no API key is set. Each pattern is one known story. The story with the most matching symptom keywords is turned into nodes and edges.

When `OPENAI_API_KEY` is set, those patterns are not sent to the model. The model reads the ticket text plus the allowed names and returns the nodes and edges itself.

Names must be spelled the same in every list. `MERGE` treats a different spelling as a different node.

| Id | Class | Symptoms | Root cause |
| --- | --- | --- | --- |
| T1 | Authentication | login_failed, password_reset | Password hash not updated in cache |
| T2 | Authentication | session_expired, login_failed | Session token TTL too short |
| T3 | Authentication | login_failed, email_not_sent | SMTP credentials expired |
| T4 | Payment | double_charge | Missing idempotency key |
| T5 | Payment | payment_failed | Payment SDK failed to load |
| T6 | Payment | payment_pending | Bank webhook missing |
| T7 | Performance | slow_page, high_cpu | Missing database index |
| T8 | Performance | timeout, slow_page | Full table scan on search |
| T9 | Data | stale_data, sync_failed | Cache not invalidated |
| T10 | Data | missing_rows, sync_failed | Nightly ETL job failed |

The 3 new tickets in `data/new_tickets.json` have `expected_category` and `expected_cause`. `main.py` prints those after the answer as a study check. They are not an input to classification, the graph, or the model.

## How a build writes one ticket

`extract_ticket` reads T3 and returns nodes and edges. With an API key, the language model writes that JSON. With no key, the hint pattern whose symptoms match the text is used instead. `build_graph` then sends Cypher like this:

```cypher
MERGE (t:Ticket {id: "T3"})
SET t.text = "Cannot login because the reset email never arrives."

MATCH (cat:Category {name: "Authentication"})
MATCH (comp:Component {name: "Email Service"})
MATCH (rc:RootCause {name: "SMTP credentials expired"})
MERGE (t)-[:ABOUT]->(cat)
MERGE (t)-[:TOUCHES]->(comp)
MERGE (t)-[:HAS_CAUSE]->(rc)

MATCH (s:Symptom {name: "email_not_sent"})
MERGE (t)-[:MENTIONS]->(s)
MERGE (s)-[:POINTS_TO]->(rc)
```

`build` deletes every node first (`MATCH (n) DETACH DELETE n`), then writes the extracted nodes and edges for each ticket. Re-running `build` gives the same graph when the tickets and the extractor stay the same.

After a build the counts are:

| Nodes | Count | Relationships | Count |
| --- | --- | --- | --- |
| Category | 4 | ABOUT | 10 |
| Component | 8 | TOUCHES | 10 |
| Symptom | 13 | MENTIONS | 17 |
| RootCause | 10 | HAS_CAUSE | 10 |
| Solution | 10 | LIVES_IN | 10 |
| Ticket | 10 | SOLVED_BY | 10 |
| | | POINTS_TO | 17 |

## How a run moves, step by step

| Step | Code | Input | Output |
| --- | --- | --- | --- |
| 0 | `extract_ticket` then `build_graph` | one ticket text + `supporting.json` | nodes and arrows in Neo4j |
| 1 | `classify_ticket` | new sentence + Naive Bayes model | one class, plus symptom names |
| 2a | `similar_tickets` | symptom names | up to 3 past tickets, scored by shared symptoms |
| 2b | `causes_for_symptoms` | symptom names | each symptom -> cause -> solution link |
| 3 | `generate` | the sentence + step 1 + step 2 | category, root cause, solution, and why |

Step 1 is multiclass: four classes, one winner. `train_category_model` fits Naive Bayes on the 10 past tickets. Each word count is a clue, and `predict_proba` returns a probability for every class. The largest probability wins. Symptom names are still found by keywords, and a ticket can have several. The graph wants that list, because the combination is what separates T1 from T3.

Step 2 does not trust the keyword class as a filter. Shared symptoms are the evidence. The past ticket's own class is returned with the row, and step 3 can use it.

Similarity is not embeddings and not "these two sentences look alike". It is a count of shared neighbors:

```cypher
MATCH (t:Ticket)-[:MENTIONS]->(s:Symptom)
WHERE s.name IN $symptoms
```

`$symptoms` is the Python list from step 1. `count(DISTINCT s)` is the score. The query then keeps walking to the cause and the solution so each row is already a full story.

Read that query one line at a time:

1. `MATCH (t:Ticket)-[:MENTIONS]->(s:Symptom)` finds tickets pointing at symptoms.
2. `WHERE s.name IN $symptoms` keeps only the symptoms found in the new ticket.
3. The next `MATCH` lines walk ticket -> cause -> solution, and ticket -> category, and ticket -> component.
4. `RETURN` picks the fields we want to print and to show the model.
5. `collect` and `count` add up the shared symptoms per ticket.
6. `ORDER BY shared_symptoms DESC` puts the best history first. `LIMIT 3` keeps the prompt small.

The second query is the distilled knowledge, not the history:

```cypher
MATCH (s:Symptom)-[:POINTS_TO]->(rc:RootCause)-[:SOLVED_BY]->(sol:Solution)
WHERE s.name IN $symptoms
```

One symptom can return several causes. That is expected. The similar-ticket score is how we choose.

## Full trace of ticket N1

Input sentence:

```
I cannot login and the reset email never comes to my inbox.
```

### Step 1 input and output

Input is that sentence plus the model trained on the 10 past tickets.

Output:

```
Authentication 0.95
Data           0.01
Payment        0.02
Performance    0.02
WINNER: Authentication

symptom login_failed    hits ['cannot login']
symptom email_not_sent  hits ['email never', 'reset email']
```

The class probabilities come from Naive Bayes. Words such as "login" and "email" also appear in the Authentication training tickets, so that class gets almost all of the probability. The symptom lines are still keyword matches. `email never` matches because the sentence says "email never comes". Those symptom names, not the 0.95, are what step 2 sends to Neo4j.

### Step 2 input and output

Input symptoms: `["login_failed", "email_not_sent"]`

Similar tickets, best first:

```
T3 shared 2 symptoms ['email_not_sent', 'login_failed']
   cause: SMTP credentials expired
   solution: Put the new mail password in the email service config
             and resend the queued reset emails.

T1 shared 1 symptom ['login_failed']
   cause: Password hash not updated in cache

T2 shared 1 symptom ['login_failed']
   cause: Session token TTL too short
```

Cause links from `POINTS_TO`:

```
email_not_sent -> SMTP credentials expired
login_failed   -> Password hash not updated in cache
login_failed   -> SMTP credentials expired
login_failed   -> Session token TTL too short
```

`login_failed` alone would be three possible causes. Adding `email_not_sent` makes T3 the only ticket with 2 shared symptoms.

### Step 3 input and output

Input is the prompt. It contains the new sentence, the keyword winner, the symptom list, the three past tickets, and the four cause links. The script prints the full prompt every time.

With no API key the mode is `graph-only`. Output:

```
CATEGORY: Authentication
ROOT CAUSE: SMTP credentials expired. The email service can no longer log in to the mail server, so reset emails sit in the queue.
SOLUTION: Put the new mail password in the email service config and resend the queued reset emails.
WHY: Past ticket T3 shares 2 symptom(s): ['email_not_sent', 'login_failed'].
```

That answer is the top graph row, written into the four fields. The study note printed afterward says the expected cause is `SMTP credentials expired`. The study note was not an input.

With an API key the mode is `llm`. The same prompt is sent to the model. The model must answer with exactly these four lines:

```
CATEGORY: ...
ROOT CAUSE: ...
SOLUTION: ...
WHY: ...
```

The instruction in the prompt says to use only the graph context and to prefer the highest `shared_symptoms`. The model writes the sentences. The graph supplies the facts.

## The other two new tickets

N2 input: `The customer was charged two times for one shirt order.`

- Step 1 winner: Payment, with probability 0.95. The word "charged" is common in the Payment training tickets.
- Symptoms: `double_charge`.
- Step 2 top ticket: T4, shared 1, cause `Missing idempotency key`.
- Step 3: refund the second charge and save an idempotency key.

N3 input: `Search page times out and feels very slow.`

- Step 1 winner: Performance, with probability 0.92. "slow" and "times" appear in the Performance training tickets.
- Symptoms: `slow_page`, `timeout`.
- Step 2 is the interesting case. `slow_page` points at two causes, because both T7 (dashboard) and T8 (search) are slow.

```
T8 shared 2 symptoms ['slow_page', 'timeout']  -> Full table scan on search
T7 shared 1 symptom  ['slow_page']             -> Missing database index
```

`timeout` is what separates them. Step 3 follows T8: add a full text index on product name.

## What the model is doing, and what it is not doing

The language model does not choose the class and it does not query Neo4j. Naive Bayes already chose the class, and Neo4j already returned the rows. `llm_gen.py` pastes those rows into the prompt. That is "graph info as context".

`ask_llm` is a short chat call: one user message, temperature 0, the model name from settings. Point `OPENAI_BASE_URL` at OpenAI, Groq, or Ollama and the function stays the same.

`parse_answer` reads the four lines back. The raw model text is printed as well, so you can see the wording before it is split.

## See the same graph in the browser

1. Open http://localhost:7474 and connect.
2. Open `cypher/examples.cypher`.
3. Run the path query:

```cypher
MATCH path = (s:Symptom {name: "email_not_sent"})-[:POINTS_TO]->(:RootCause)-[:SOLVED_BY]->(:Solution)
RETURN path
```

The browser draws the nodes and the arrows. Query 5 in that file is the N1 similarity search with the symptom list written in place of `$symptoms`.

`python main.py show` prints the same facts as text, including every arrow.

## Try a change

1. Add a closed ticket to `data/tickets.json` with an `id` and a `text` that includes the complaint, the cause, and the fix.
2. If you have no API key, add a matching story to `patterns` in `data/supporting.json`. With an API key, the model extracts the nodes and edges from the text.
3. Run `python main.py build` so Neo4j is replaced with the new graph.
4. Run `python main.py analyze "your new sentence"`.

Watch step 2. If the new sentence shares symptoms with the ticket you added, that ticket shows up in the output and in the prompt.

## Left out on purpose

This is a teaching sketch of the pipeline you described: past history, graph similarity, then a model.

It does not have login screens, queues, big data, vector embeddings, or error handling. The class is a small Naive Bayes model on 10 sentences. Symptom names are still keywords, because each symptom has too few examples for its own model. The graph walk behind that stays the same. That split is the thing to notice.
