CREATE DATABASE nyt_pipeline;

USE nyt_pipeline;

CREATE TABLE nyt_staging (
    id INT AUTO_INCREMENT PRIMARY KEY,
    article_id VARCHAR(255) UNIQUE,
    title TEXT NOT NULL,
    author TEXT, 
    published_date DATETIME NOT NULL,
    url TEXT NOT NULL,
    section VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    
CREATE TABLE nyt_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    article_id VARCHAR(255) UNIQUE,
    title TEXT NOT NULL,
    author TEXT,
    published_date DATETIME NOT NULL,
    url TEXT NOT NULL,
    section VARCHAR(255),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP);
    
DESC nyt_staging;

DESC nyt_articles;

SELECT COUNT(*) FROM nyt_articles;

SELECT * FROM nyt_articles ORDER BY id asc;

SELECT * FROM nyt_staging LIMIT 10;

SELECT * FROM nyt_articles LIMIT 10;

SELECT section, COUNT(*) 
FROM nyt_articles 
GROUP BY section 
ORDER BY COUNT(*) DESC;

SELECT * FROM nyt_articles ORDER BY published_date DESC LIMIT 10;

SELECT * FROM nyt_articles ORDER BY last_updated DESC LIMIT 10;




















