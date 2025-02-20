NYT Data Pipeline

Overview

This project is an automated data pipeline that fetches top news articles from the New York Times API, 
processes the data, and stores it in a MySQL database. The pipeline runs daily at 5 PM using Windows 
Task Scheduler and ensures data integrity through deduplication and error handling.

Features

Fetches top news articles from the New York Times API
Stores data in a MySQL database using a staging and final table approach
Prevents duplicate entries with efficient SQL queries
Logs execution details in nyt_pipeline.log for monitoring and debugging
Sends Slack notifications for pipeline failures
Fully automated execution using Windows Task Scheduler

Project Structure

nyt_pipeline.py - Main Python script for the pipeline
nyt_pipeline.sql - SQL schema and queries
nyt_pipeline.log - Log file (auto-generated)
507 Final Project.ipynb - Jupyter notebook for testing/debugging
README.md - Project documentation

Setup & Installation

Clone the Repository
git clone https://github.com/tshehadey/Projects.git
cd Projects/SQL_NYT_Pipeline

Configure Environment Variables

Set up the following environment variables:
MYSQL_PASSWORD=your_mysql_password
NYT_API_KEY=your_nyt_api_key
SLACK_WEBHOOK_URL=your_slack_webhook_url
For Windows, use set instead of export.

Create MySQL Database & Tables

Run the SQL script to set up the database:
mysql -u root -p < nyt_pipeline.sql

Schedule the Pipeline to Run Daily

Use Windows Task Scheduler to run the pipeline every day at 5 PM.
Create a new task and set the following command:
python path/to/nyt_pipeline.py

Run the Pipeline Manually (If Needed)
python nyt_pipeline.py

Monitoring and Debugging with Log Files

The pipeline maintains a log file called nyt_pipeline.log to track execution details, errors, and warnings. This helps with debugging and monitoring pipeline performance.

To view the log, simply navigate to the folder where you pipeline is store and open the .log file. 
Check the latest entries at the bottom for timestamps and status updates.
INFO: Indicates normal pipeline execution steps, such as "Fetched 20 new articles."
ERROR: Logs critical failures that may require manual intervention, such as "Database connection failed" or "Duplicate entry detected."

Troubleshooting

Duplicate Entry Errors in MySQL:

The pipeline prevents duplicates, but errors may still occur.
Fix: Modify nyt_pipeline.sql to use INSERT IGNORE or REPLACE INTO.

Slack Notifications Not Working:

Check if SLACK_WEBHOOK_URL is correctly set in the environment variables.
Ensure Slack is reachable from the server running the script.

Future Improvements

Optimize SQL queries with indexing for faster lookups
Store historical article data for trend analysis
Build a Flask-based dashboard to monitor fetched articles
Implement retry logic for API requests in case of failures
