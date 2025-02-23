import logging
import requests
import mysql.connector
import os
import json

# Explicit log file path
LOG_FILE_PATH = r"C:\Users\tysir\nyt_pipeline\nyt_pipeline.log"

# Setup logging to write to file and console
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",  # Append mode (keeps previous logs)
    force=True  # Ensures reloading of logging handlers
)

# Add logging to output
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Load environment variables
password = os.getenv("MYSQL_PASSWORD")
api_key = os.getenv("NYT_API_KEY")
slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_alert(message):
    """Send an alert to Slack in case of failure."""
    if slack_webhook_url:
        payload = {"text": f":warning: *NYT Pipeline Failure Alert* :warning:\n\n{message}"}
        response = requests.post(slack_webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            logging.error(f"Failed to send Slack alert: {response.text}")

logging.info("Pipeline execution started.")

try:
    # Connect to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password=password,
        database="nyt_pipeline")

    logging.info("Connected to MySQL successfully.")
    cursor = conn.cursor()

    # Fetch Data from NYT API
    URL = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={api_key}"

    response = requests.get(URL)
    response.raise_for_status()  # Raise an error for failed requests

    data = response.json()
    logging.info(f"Fetched {len(data['results'])} articles successfully.")

    # Extract Relevant Data
    articles = []
    for article in data['results']:
        article_id = article.get('uri', 'N/A').split('/')[-1]
        title = article.get('title', 'No Title')
        author = article.get('byline', '').replace("By ", "").strip()
        published_date = article.get('published_date', None)
        url = article.get('url', 'No URL')
        section = article.get('section', 'No Section')

        articles.append((article_id, title, author, published_date, url, section))

    logging.info(f"Extracted {len(articles)} articles.")

    # Insert into Staging Table
    insert_query = """
    INSERT INTO nyt_staging (article_id, title, author, published_date, url, section)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        title = VALUES(title),
        author = VALUES(author),
        published_date = VALUES(published_date),
        url = VALUES(url),
        section = VALUES(section);
    """

    cursor.executemany(insert_query, articles)
    conn.commit()
    logging.info("Data inserted into nyt_staging successfully.")

    # Move Data to Final Table (includes deduplication)
    insert_final_query = """
	INSERT INTO nyt_articles (article_id, title, author, published_date, url, section)
	SELECT s.article_id, s.title, s.author, s.published_date, s.url, s.section
	FROM nyt_staging s
	WHERE NOT EXISTS (
    SELECT 1 FROM nyt_articles a WHERE a.article_id = s.article_id);
"""

    cursor.execute(insert_final_query)
    conn.commit()
    logging.info("Data moved to nyt_articles successfully.")

    # Close Connection
    cursor.close()
    conn.close()
    logging.info("Pipeline execution completed successfully.")

except Exception as e:
    error_message = f"Pipeline failed with error: {e}"
    logging.error(error_message)
    send_slack_alert(error_message)

# Ensure logs are written immediately (fixes Task Scheduler issues)
logging.shutdown()
