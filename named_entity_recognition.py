
import spacy

# Load the pretrained English NLP model
nlp = spacy.load("en_core_web_sm")

# Input sentence from the assignment
sentence = (
    " LeBron James as the NBA player of the United States "
    "in Laker and won the NBA Final MVP in 2020."
)

# Process the sentence
doc = nlp(sentence)

# Print named entities and character positions
for ent in doc.ents:
    print(
        f"Entity: {ent.text} | "
        f"Label: {ent.label_} | "
        f"Start: {ent.start_char} | "
        f"End: {ent.end_char}"
    )
