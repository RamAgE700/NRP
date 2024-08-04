import requests
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import spacy

# Function to fetch news article
def fetch_news_article(api_key):
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])

    if not articles:
        raise ValueError("No articles found")

    # Select the first article with content
    for article in articles:
        content = article.get('content')
        if content:
            return article['title'], content

    raise ValueError("No articles with content found")

# Fetching the article
api_key = '30c5a42402a64487a66fff798d5ad896'
try:
    title, content = fetch_news_article(api_key)
    print(f"Title: {title}\n")
    print(f"Content: {content}\n")
except ValueError as e:
    print(e)
    content = "No content available for analysis."

# NLTK NER Extraction
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def nltk_ner(text):
    entities = []
    for sent in nltk.sent_tokenize(text):
        for chunk in ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, Tree):
                entity = " ".join(c[0] for c in chunk)
                entities.append((entity, chunk.label()))
    return entities

nltk_entities = nltk_ner(content)
print("NLTK Named Entities:")
for entity in nltk_entities:
    print(entity)

# SpaCy NER Extraction
nlp = spacy.load('en_core_web_sm')

def spacy_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

spacy_entities = spacy_ner(content)
print("\nSpaCy Named Entities:")
for entity in spacy_entities:
    print(entity)

# Comparing Results
def compare_entities(nltk_entities, spacy_entities):
    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)

    print("\nEntities detected by both NLTK and SpaCy:")
    print(nltk_set & spacy_set)

    print("\nEntities detected only by NLTK:")
    print(nltk_set - spacy_set)

    print("\nEntities detected only by SpaCy:")
    print(spacy_set - nltk_set)

compare_entities(nltk_entities, spacy_entities)
