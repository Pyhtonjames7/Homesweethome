import feedparser

def fetch_legal_news(url):
    # Parse the RSS feed
    feed = feedparser.parse(url)
    
    # Check if the feed was successfully parsed
    if feed.bozo == 1:
        print("Failed to fetch or parse the feed.")
        return
    
    print("Legal News:")
    for entry in feed.entries[:5]:  # Let's limit to the first 5 entries for brevity
        print(f"Title: {entry.title}")
        print(f"Link: {entry.link}")
        print(f"Published: {entry.published}")
        print("-" * 50)

# Example RSS feed URL (You should replace this with the actual RSS feed URL you intend to use)
rss_url = 'https://www.law.com/newyorklawjournal/rss/'
fetch_legal_news(rss_url)