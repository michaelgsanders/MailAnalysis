import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
from nltk.util import ngrams
from collections import Counter
import re

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

"""

Mail Analysis - a tool that uses gmail api to analyze your sent messages 
for speech patterns and commonly used words or phrases

Developed by Michael Sanders, Cornell University 2026, mgs264@cornell.edu

"""

def main():
    """
    Lists the user's Gmail labels and fetches the sent messages.
    """
    nltk.data.path.append('/Users/michaelsanders/Desktop/nltk_data') # manually path for nltk data
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        sent_texts = readSentMail(service)
        analyze_texts(sent_texts)

    except HttpError as error:
        print(f"An error occurred: {error}")

def readSentMail(service):
    """Fetches and stores the text of all sent messages."""
    try:
        sent_texts = []
        results = service.users().messages().list(userId='me', labelIds=['SENT'], maxResults=500).execute()
        messages = results.get('messages', [])

        while 'nextPageToken' in results:
            page_token = results['nextPageToken']
            results = service.users().messages().list(userId='me', labelIds=['SENT'], maxResults=500, pageToken=page_token).execute()
            messages.extend(results.get('messages', []))

        if not messages:
            print("No sent messages found.")
            return []

        for msg in messages:
            msg_id = msg['id']
            msg_data = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            payload = msg_data['payload']

            # recursively extract the message parts
            def extract_parts(parts, sent_texts):
                for part in parts:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        text = part['body']['data']
                        decoded_text = base64.urlsafe_b64decode(text).decode('utf-8')
                        sent_texts.append(decoded_text)
                    elif 'parts' in part:
                        extract_parts(part['parts'], sent_texts)

            # Check if payload has 'parts'
            if 'parts' in payload:
                extract_parts(payload['parts'], sent_texts)
            elif 'body' in payload and 'data' in payload['body']:
                # Handle the case where there's no 'parts', just 'body'
                text = payload['body']['data']
                decoded_text = base64.urlsafe_b64decode(text).decode('utf-8')
                sent_texts.append(decoded_text)

        print(f"Total sent messages stored in the list: {len(sent_texts)}")
        return sent_texts

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def analyze_texts(texts):
    """Analyzes the given list of texts for speech patterns."""
    all_words = []
    all_text = ' '.join(texts)
    
    words = word_tokenize(all_text)
    words = [word.lower() for word in words if word.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    fdist = FreqDist(words)
    
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)
    
    bigrams = ngrams(words, 2)
    bigram_freq = Counter(bigrams)
    
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(all_text)
    
    print("Most common words:")
    for word, frequency in fdist.most_common(10):
        print(f'{word}: {frequency}')
    
    print("\nMost common bigrams:")
    for bigram, frequency in bigram_freq.most_common(10):
        print(f'{" ".join(bigram)}: {frequency}')
    
    print("\nSentiment analysis:")
    for k, v in sentiment.items():
        print(f'{k}: {v}')
    
    print("\nNamed Entities:")
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            print(f'{chunk.label()}: {" ".join(c[0] for c in chunk)}')

if __name__ == "__main__":
    main()
