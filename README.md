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
2. Scrape some data into the database: `mtsearch.py update -p`
3. Perform searches: `mtsearch.py search "your keyword"`

### Configuration File

The script uses a configuration file (`config.json`) with the following fields:

- `api_key`: Your API key.

- `domain`: The URL of the M-Team site. Leave this empty to use the default domain.

- `request_interval`: The time interval (in seconds) between each API request. Set to `0` to make requests without any delay.

- `hourly_limit`: The maximum number of requests permitted per hour. Set to `0` for no limit.

- `nordvpn_path`: The file path to the NordVPN executable, used to manage IP rotation and bypass throttling. Ensure the NordVPN client is installed. Common paths include:
  - **Windows**: `C:\Program Files\NordVPN\nordvpn.exe`
  - **Linux**: `nordvpn`

- `search_params`: A list of parameters for the `/api/torrent/search` API used during the execution of `mtsearch.py update -p`. These parameters determine the scope and type of data retrieved. For a complete list of available parameters, consult the official M-Team API documentation.
  Common parameters include:
  - `mode`: Determines the type of content to search. Available values are `normal`, `adult`, `movie`, `music`, `tvshow`, `waterfall`, `rss`, `rankings`.
  - `categories`: An array of integers that identify specific categories within a selected mode. An empty array includes all categories.

Example Configuration:

```json
{
    "api_key": "your_api_key_here",
    "domain": "",
    "request_interval": 10,
    "hourly_limit": 0,
    "nordvpn_path": "nordvpn",
    "search_params": [
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

### Command-Line Usage

#### For searching

```
usage: mtsearch.py search [-h] [-e | -f | -m] [pattern]

positional arguments:
  pattern      specify the search pattern

options:
  -h, --help   show this help message and exit
  -e, --regex  use regular expression matching
  -f, --fixed  use fixed-string FTS5 matching (default)
  -m, --fts    use freeform FTS5 matching
```

- Enter interactive search mode (use -e, -f, -m to specify search modes):

  `mtsearch.py search` or `mtsearch.py s`

- Search for a specific keyword (equivalent to using -f):

  `mtsearch.py search "foo"`

- Search using [FTS5](https://www.sqlite.org/fts5.html) syntax (without -m, the 'OR' operator is treated literally):

  `mtsearch.py search -m "foo OR bar"`

- Search using a regular expression (e.g., matches 2022, 2023, 2024):

  `mtsearch.py search -e "202[2-4]"`

#### For updating

```
usage: mtsearch.py update [-h] [-c CACHE_DIR] [--no-limit] (-p [PAGES] | -i ID [ID ...] | --recreate)

options:
  -h, --help      show this help message and exit
  -c CACHE_DIR    save torrent files to this directory
  --no-limit      temporarily disable rate limiting

actions:
  -p [PAGES]      scrape one or more pages in 'page' or 'start-end' format (default: 1-3)
  -i ID [ID ...]  update one or more torrent IDs
  --recreate      recreate the database
```

- Scrape the 5 most recent pages, bypassing the rate limiter.

  `mtsearch.py update -p 1-5 --no-limit`

- Scrape torrent ID 3, 5, and 7.

  `mtsearch.py update -i 3 5 7`

## Data File

A SQLite database named `data.db` will be created in the script's directory, storing all scraped torrent data. Ensure you back up this database as needed.

## API throttling:

> **Note**: Currently, M-Team appears to implement API throttling dynamically. You're free to experiment with different settings of rate limiting until you get banned. For scraping a small number of pages, use the `--no-limit` switch to temporarily disable the rate limiter.

## Authors

- David Pi
- ChatGPT by OpenAI
