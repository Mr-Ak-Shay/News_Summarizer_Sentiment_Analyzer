import requests
from transformers import pipeline
from textblob import TextBlob

def fetch(api_key, query, language="en", page_size=5):
    """
    Fetches live News based on the particular query given!
    """
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        raise Exception(f"Error fetching news: {response.json()}")

def summarize(article_content):
    """
    Summarizes the content of a news article.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(article_content, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def sentiment(text):
    """
    Analysis using sentiment of the text using TextBlob.
    """
    analysis = TextBlob(text)
    sentiment = analysis.sentiment
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }

def main():
    api = "" #Enter your NewsAPI Token
    query = "artificial intelligence" #You can enter you desired query/topic
    try:
        articles = fetch(api, query,page_size=10)
        print(f"Found these {len(articles)} articles on '{query}'")

        for idx, article in enumerate(articles, start=1):
            title = article.get("title", "No title")
            content = article.get("description", "No description")
            url = article.get("url", "No url")

            print(f"Article {idx}: {title}")
            print(f"URL: {url}")

            if content and len(content.split()) > 20:
                summary = summarize(content)
                print("\nSummary:")
                print(summary)

                sentiment_result = sentiment(summary)
                print("\nSentiment Analysis: ")
                print(f"Polarity: {sentiment_result['polarity']}(negative to positive)")
                print(f"Subjectivity: {sentiment_result['subjectivity']}(objective to subjective)")
            else:
                print("\nContent is too short to be summarized!")

            print("\n" + "-" * 50 + "\n")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
