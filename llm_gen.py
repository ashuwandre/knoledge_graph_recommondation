"""
Step 3. Write the root cause and the solution.

The graph already found the facts. This file turns those facts into
a short answer a support agent can read.

Two modes:
    graph-only: no API key. Copy the past ticket with the highest share count.
    llm: an API key is set. The model writes the four lines, using the
         graph rows as the only allowed context.

Input: ticket text, classifier result, similar-ticket rows, cause rows
Output: dict with mode, prompt, raw model text, and the parsed four lines
"""


def similar_to_text(rows):
    """
    Turn similar-ticket rows into plain lines for the prompt.

    Input: list of dicts from similar_tickets()
    Output: one multiline string. "(no similar ticket)" when the list is empty.
    """
    if not rows:
        return "(no similar ticket)"
    lines = []
    for row in rows:
        lines.append(
            f"- ticket {row['ticket_id']} | shared_symptoms {row['shared_symptoms']} "
            f"| matched {row['matched_symptoms']} | category {row['category']} "
            f"| component {row['component']} | cause {row['root_cause']} "
            f"| detail {row['cause_detail']} | solution {row['solution_steps']}"
        )
    return "\n".join(lines)


def causes_to_text(rows):
    """
    Turn symptom-to-cause rows into plain lines for the prompt.

    Input: list of dicts from causes_for_symptoms()
    Output: one multiline string. "(no cause link)" when the list is empty.
    """
    if not rows:
        return "(no cause link)"
    lines = []
    for row in rows:
        lines.append(
            f"- symptom {row['symptom']} -> cause {row['root_cause']} "
            f"| component {row['component']} | solution {row['solution_steps']}"
        )
    return "\n".join(lines)


def build_prompt(ticket_text, classification, similar_rows, cause_rows):
    """
    Build the text we send to the model.

    Why the prompt is strict:
        The model is good at sentences and weak at remembering your old tickets.
        The graph rows are pasted in so the model writes from those facts.

    Input:
        ticket_text: the new complaint
        classification: output of classify_ticket()
        similar_rows: output of similar_tickets()
        cause_rows: output of causes_for_symptoms()

    Output:
        one string, the full prompt
    """
    symptom_names = [item["name"] for item in classification["symptoms"]]
    winner = classification["winner"]

    return f"""You are a support analyst. Write a root cause and a solution for the new ticket.
Use only the knowledge graph context below. Prefer the past ticket with the highest shared_symptoms.

New ticket:
{ticket_text}

ML classifier (Naive Bayes):
category {winner["name"]} with probability {winner["score"]}

Symptoms:
{symptom_names}

Similar past tickets from Neo4j:
{similar_to_text(similar_rows)}

Symptom-to-cause links from Neo4j:
{causes_to_text(cause_rows)}

Reply with exactly these four lines and nothing else:
CATEGORY: one of Authentication, Payment, Performance, Data
ROOT CAUSE: one or two sentences
SOLUTION: one or two sentences
WHY: which past ticket or graph link you used
"""


def answer_from_graph(classification, similar_rows):
    """
    Produce the four answer lines without calling a model.

    Why this exists:
        You can learn the graph before you have an API key.
        The best row is already first, because the Cypher query sorts
        by shared_symptoms descending.

    Input: classifier result and similar-ticket rows
    Output: dict with CATEGORY, ROOT CAUSE, SOLUTION, WHY
    """
    if not similar_rows:
        return {
            "CATEGORY": classification["winner"]["name"],
            "ROOT CAUSE": "No similar past ticket was found in the graph.",
            "SOLUTION": "No stored solution for these symptoms.",
            "WHY": "None of the symptom nodes are shared with a past ticket.",
        }

    best = similar_rows[0]
    return {
        "CATEGORY": best["category"],
        "ROOT CAUSE": best["root_cause"] + ". " + best["cause_detail"],
        "SOLUTION": best["solution_steps"],
        "WHY": (
            f"Past ticket {best['ticket_id']} shares {best['shared_symptoms']} "
            f"symptom(s): {best['matched_symptoms']}."
        ),
    }


def parse_answer(text):
    """
    Pull the four requested lines out of the model reply.

    Input: raw model text
    Output: dict with CATEGORY, ROOT CAUSE, SOLUTION, WHY.
        A missing line stays as an empty string. The raw text is still
        printed by main.py so you can read whatever the model wrote.
    """
    parsed = {
        "CATEGORY": "",
        "ROOT CAUSE": "",
        "SOLUTION": "",
        "WHY": "",
    }
    for line in text.splitlines():
        clean = line.strip()
        for key in parsed:
            if clean.upper().startswith(key + ":"):
                parsed[key] = clean.split(":", 1)[1].strip()
    return parsed


def ask_llm(settings, prompt):
    """
    Send the prompt to a chat model and return the text it writes.

    The OpenAI client also works with any server that speaks the same
    API, including a local Ollama server. Change base_url and model
    in .env. The Python in this function stays the same.

    Input: settings dict and the prompt string
    Output: the model text (a string)
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
    response = client.chat.completions.create(
        model=settings["model"],
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate(settings, ticket_text, classification, similar_rows, cause_rows):
    """
    Run step 3 in graph-only mode or in model mode.

    Input:
        settings: from config.get_settings()
        ticket_text, classification, similar_rows, cause_rows: earlier steps

    Output:
        dict with:
            mode: "graph-only" or "llm"
            prompt: the exact text that would be (or was) sent to the model
            raw: model text, or "" in graph-only mode
            parsed: the four answer lines
    """
    prompt = build_prompt(ticket_text, classification, similar_rows, cause_rows)

    # Empty key means we stay on the graph result so the lesson still runs.
    if not settings["api_key"]:
        return {
            "mode": "graph-only",
            "prompt": prompt,
            "raw": "",
            "parsed": answer_from_graph(classification, similar_rows),
        }

    raw = ask_llm(settings, prompt)
    return {
        "mode": "llm",
        "prompt": prompt,
        "raw": raw,
        "parsed": parse_answer(raw),
    }
