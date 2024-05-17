# MTSearch - A M-Team Scraper and Search Utility

## Description

MTSearch is a powerful utility for scraping torrent data from M-Team torrent sites and storing it in a local database for fast searching. It supports various search modes including fixed-string matching, SQLite FTS5 matching, and regular expression matching.

### Features
- **Automatic Scraping**: Scrape torrent data across a range of pages as specified.
- **Rate Limiter**: Avoid being banned with a configurable rate limiter.
- **Full-Text Search**: Perform searches across page titles, torrent names, and file paths inside torrent files.
- **Blazing fast**: Utilizes SQLite FTS5 for rapid searches (milliseconds response for over 250k torrents and 25m paths).
- **Advanced Search**: Use multi-processing for regular expression searching. Slower but more powerful and comprehensive.

## Installation

To get started, you'll need Python 3. Then clone the GitHub repository and install the required packages:

```bash
git clone https://github.com/libertypi/mtsearch.git

cd mtsearch
pip install -r requirements.txt
```

## Usage

1. Upon the first run, a default configuration file (`config.json`) will be generated alongside the script. Edit this config file before running this script again.
2. Scrape some data into the database.
3. Perform searches.

### Configuration File

The script uses a configuration file (`config.json`) with the following fields:

- `api_key`: API key for M-Team site.
- `domain`: The URL of the M-Team site. Leave empty to use the default domain.
- `hourly_limit`: Maximum number of requests in an hour. Set to 0 to disable.
- `request_interval`: Time interval between each request in seconds. Set to 0 to disable.
- `mode_categories`: List of modes and their subcategories to scrape. `mode` can be: `normal`, `adult`, `movie`, `music`, `tvshow`, `waterfall`, `rss`, `rankings`. `categories` is a list of integers where an empty list includes all subcategories.

Example Configuration:

```json
{
    "api_key": "your_api_key_here",
    "domain": null,
    "hourly_limit": 100,
    "request_interval": 10,
    "mode_categories": [
        {
            "mode": "adult",
            "categories": []
        },
        {
            "mode": "movie",
            "categories": [404, 421]
        }
    ]
}
```

### Basic Command-Line Usage

```
mtsearch.py [-h] [-s | -u] [-e | -f | -m] [-r PAGE_RANGE] [--dump DUMP_DIR] [--no-limit] [pattern]

Operation Modes:
  -h, --help       show this help message and exit
  -s, --search     Search for a pattern in the database (default)
  -u, --update     Update the database

Search Options:
  -e, --regex      Use regular expression matching
  -f, --fixed      Use fixed-string matching (default)
  -m, --fts        Use freeform SQLite FTS5 matching
  pattern          Specify the search pattern

Update options:
  -r PAGE_RANGE    Specify page range as 'start-end', or 'end' (Default: 3)
  --no-limit       Disable rate limit defined in the config file
```

### Examples (For searching)

- Enter interactive search mode (use -e, -f, -m to specify search modes):

  `mtsearch.py`

- Search for a specific keyword (equivalent to using -f):

  `mtsearch.py "foo"`

- Search using [FTS5](https://www.sqlite.org/fts5.html) syntax (without -m, the 'OR' operator is treated literally):

  `mtsearch.py -m "foo OR bar"`

- Search using a regular expression (e.g., matches 2022, 2023, 2024):

  `mtsearch.py -e "202[2-4]"`

### Examples (For updating)

- Scrape the 5 most recent pages, bypassing the rate limiter.

  `mtsearch.py -u --no-limit -r 5`

## Data File

A SQLite database named `data.db` will be created in the script's directory, storing all scraped torrent data. Ensure you back up this database as needed.

## Notes:

Currently, M-Team's website appears to implement two types of rate limiting:

1. **Per-Request Interval**: Making 6 consecutive requests without a 20-second gap between each will result in a 120-second ban.
2. **Global Limit**: There is a maximum number of requests allowed within a specific time window. Exceeding this limit will result in a 1-day ban.

In this script, both types of rate limiting are addressed through the configuration file:

- `"hourly_limit": 100` signifies that a maximum of 100 requests can be made within a 3600-second window.
- `"request_interval": 20` implies that there should be a 20-second gap between each request.

> **Note**: You're free to experiment with different settings of rate limits until you get banned. For scraping a small number of pages, use the `--no-limit` switch to temporarily disable the rate limiter.

## Authors

- David Pi
- ChatGPT by OpenAI
