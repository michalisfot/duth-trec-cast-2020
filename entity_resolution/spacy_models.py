import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
print([(w.text, w.pos_) for w in doc])
