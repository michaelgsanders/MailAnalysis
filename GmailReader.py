import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import re

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

"""

Mail Analysis - a tool that uses gmail api to analyze your sent messages 
for speech patterns and commonly used words or phrases

Developed by Michael Sanders, Cornell University 2026, mgs264@cornell.edu

"""


def main():
    """
    Lists the user's Gmail labels and fetches the first 100 sent messages.
    """
    nltk.data.path.append('/Users/michaelsanders/Desktop/nltk_data') # manually path for nltk data
    creds = None
    # token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    # to restart authorization flow, delete token.json
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # if there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # call the gmail API
        service = build("gmail", "v1", credentials=creds)
        sent_texts = readSentMail(service)
        analyze_texts(sent_texts)

    except HttpError as error:
        # TODO - Handle errors from gmail API
        print(f"An error occurred: {error}")

def readSentMail(service):
    """Fetches and stores the text of the first 100 sent messages in a list."""
    try:
        # Fetch sent messages
        results = service.users().messages().list(userId='me', labelIds=['SENT'], maxResults=100).execute()
        messages = results.get('messages', [])

        if not messages:
            print("No sent messages found.")
            return []

        sent_texts = []
        for msg in messages:
            msg_id = msg['id']
            msg_data = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            payload = msg_data['payload']

            # check if payload has 'parts'
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        text = part['body']['data']
                        decoded_text = base64.urlsafe_b64decode(text).decode('utf-8')
                        sent_texts.append(decoded_text)
            elif 'body' in payload and 'data' in payload['body']:
                # handle the case where there's no 'parts', just 'body'
                text = payload['body']['data']
                decoded_text = base64.urlsafe_b64decode(text).decode('utf-8')
                sent_texts.append(decoded_text)

        print("First 100 sent messages stored in the list.")
        return sent_texts

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def analyze_texts(texts):
    """Analyzes the given list of texts for speech patterns.
        Currently rudimentary nlp analysis using nltk.
    """
    all_words = []
    for text in texts:
        words = word_tokenize(text)
        words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words if word.isalnum()] # regex for
        all_words.extend(words)
    
    fdist = FreqDist(all_words)
    
    print("Most common words:")
    for word, frequency in fdist.most_common(10):
        print(f'{word}: {frequency}')
    

if __name__ == "__main__":
    main()
