MTeam Scraper and Search Utility
================================

Description:
------------
This script is designed to scrape torrent data from M-Team torrent sites and
store it in a local SQLite database. It supports various search modes including
fixed-string matching, SQLite FTS5 matching, and regular expression matching.

Main Functionality:
-------------------
1. Search for torrents in the local database.
2. Update the local database by scraping new torrents from the website.

Usage:
------
- For searching: `python mtsearch.py -s "search_query"`
- For updating: `python mtsearch.py -u`

Configuration File:
-------------------
The script uses a configuration file (`config.json`) with the following fields:
- `domain`: The URL of the M-Team site (default is "https://xp.m-team.io").
- `pages`: List of pages to scrape (e.g., ["adult.php"]).
- `username`: Username for M-Team site (optional).
- `password`: Password for M-Team site (optional).
- `maxRequests`: Maximum number of requests in a time window (default is 100).
- `timeWindow`: Time window in seconds for maxRequests (default is 3600).
- `requestInterval`: Time interval between each request in seconds (default is 20).

Authors:
--------
- David Pi
- ChatGPT - OpenAI
