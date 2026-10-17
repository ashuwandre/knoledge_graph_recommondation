"""
Step 1. Turn a raw ticket sentence into one class and a few symptom names.

The class is chosen by a small machine-learning model:
    CountVectorizer turns each sentence into word counts.
    MultinomialNB (Naive Bayes) learns which words go with which class.
    The four classes are Authentication, Payment, Performance, and Data.

Training rows are read back from Neo4j after the graph is built.
Each row is a ticket sentence plus the Category connected by ABOUT.
Each past ticket already has a human label in the "category" field.

Symptom names still come from keyword matches. The graph stores nodes
named login_failed, email_not_sent, and so on. The search in step 2
needs those exact names. Most symptoms appear on only one past ticket,
which is too little data for a second model.

Input: past tickets for training, then one new sentence
Output: a probability for every class, the winning class, and symptom names
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_category_model(tickets):
    """
    Fit one multiclass model on the past tickets.

    Why:
        The new ticket has no class yet. The past tickets do.
        Naive Bayes learns a simple rule from those labeled sentences:
        some words show up more often in one class than in the others.

    Input:
        tickets: list of past ticket dicts. Each dict needs "text" and "category".

    Output:
        a fitted sklearn Pipeline.
        You can pass a new sentence to model.predict_proba and get
        one probability per class.

    How:
        CountVectorizer splits a sentence on spaces and counts each word.
        "cannot login" becomes the counts {cannot: 1, login: 1}.
        MultinomialNB.fit reads those counts together with the class label.
    """
    texts = []
    labels = []
    for ticket in tickets:
        texts.append(ticket["text"])
        labels.append(ticket["category"])

    # One object so later we can hand it a raw sentence.
    # "counts" runs first, then "nb" sees the word counts.
    model = Pipeline([
        ("counts", CountVectorizer()),
        ("nb", MultinomialNB()),
    ])
    model.fit(texts, labels)
    return model


def matched_keywords(text, keywords):
    """
    Find which keywords appear inside the ticket text.

    Why:
        Symptom nodes in Neo4j have fixed names. This list is how a
        sentence is turned into those names.

    Input:
        text: the ticket sentence
        keywords: list of phrases for one symptom

    Output:
        list of phrases that were found. Empty list means no hit.
        The check is a plain substring test on lowercase text.
    """
    lowered = text.lower()
    found = []
    for word in keywords:
        if word.lower() in lowered:
            found.append(word)
    return found


def classify_ticket(text, model, symptoms):
    """
    Predict one category and collect every symptom mentioned in the text.

    Why:
        Step 2 searches Neo4j by symptom name. The predicted class is
        also sent to the language model as a first guess.

    Input:
        text: the new ticket sentence
        model: the Pipeline returned by train_category_model
        symptoms: list of {name, keywords} from the dataset

    Output:
        dict with three keys:
            category_scores: one row per class, score is a probability from 0 to 1
            winner: the row whose class the model chose
            symptoms: symptoms whose keywords appear in the text

    The probabilities come from predict_proba. They add up to 1.
    The winning class is the one with the highest probability.
    Sklearn wants a list of sentences, so the one new ticket is wrapped
    in a list: [text].
    """
    # Block: ask the model for every class probability.
    probabilities = model.predict_proba([text])[0]
    class_names = list(model.classes_)
    category_scores = []
    for name, probability in zip(class_names, probabilities):
        category_scores.append({
            "name": name,
            "score": round(float(probability), 4),
        })

    # Block: the predicted class. predict uses the same probabilities
    # and returns the name with the largest one.
    winner_name = model.predict([text])[0]
    winner = {"name": winner_name, "score": 0.0}
    for row in category_scores:
        if row["name"] == winner_name:
            winner = row

    # Block: symptom names for the graph. This part is not the Naive Bayes
    # model. A ticket can mention more than one symptom, and the graph
    # uses that combination to separate similar causes.
    symptom_hits = []
    for symptom in symptoms:
        hits = matched_keywords(text, symptom["keywords"])
        if hits:
            symptom_hits.append({
                "name": symptom["name"],
                "hits": hits,
            })

    return {
        "category_scores": category_scores,
        "winner": winner,
        "symptoms": symptom_hits,
    }
