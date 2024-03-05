import requests
import re

def normalize_word(word):
    # Convert to lowercase and remove special characters and spaces
    normalized = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
    return normalized

def get_linked_pages_titles_with_variations(word):
    base_url = "https://en.wikipedia.org/w/api.php"
    normalized_word = normalize_word(word)

    params = {
        "action": "query",
        "format": "json",
        "titles": normalized_word,
        "prop": "links",
        "pllimit": "max"  # Retrieve all linked pages
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    linked_pages_titles = []
    if "query" in data and "pages" in data["query"]:
        page = next(iter(data["query"]["pages"].values()))
        if "links" in page:
            linked_pages_titles = [link["title"] for link in page["links"]]

    return linked_pages_titles

# Specify the word (page title) for which you want to retrieve linked pages
word = "couch"
linked_pages = get_linked_pages_titles_with_variations(word)

print("Linked Pages:")
for title in linked_pages:
    print(title)
