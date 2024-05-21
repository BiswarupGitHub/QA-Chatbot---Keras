import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_wikipedia_page(url):
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', {'class': 'mw-parser-output'})
    
    # Initialize the question variable
    question = None
    qa_pairs = []
    
    # Extract the headings and paragraphs
    for element in content_div.find_all(['h2', 'h3', 'p']):
        if element.name == 'h2' or element.name == 'h3':
            question = element.get_text(strip=True).replace('[edit]', '')
        elif element.name == 'p':
            answer = element.get_text(strip=True)
            if question and answer:
                qa_pairs.append((question, answer))
    
    return qa_pairs

def save_to_csv(qa_pairs, filename):
    df = pd.DataFrame(qa_pairs, columns=['question', 'answer'])
    df.to_csv(filename, index=False)

def main():
    url = 'https://en.wikipedia.org/wiki/Chatbot'
    qa_pairs = scrape_wikipedia_page(url)
    save_to_csv(qa_pairs, 'scrap_data.csv')
    print('Data scrapped')

if __name__ == '__main__':
    main()
