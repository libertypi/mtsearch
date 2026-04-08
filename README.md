# MTSearch - A M-Team Scraper and Search Utility

## Description

MTSearch is a powerful utility for scraping torrent data from M-Team torrent sites and storing it in a local database for fast searching. It supports various search modes including SQLite FTS5 matching and regular expression searching.

### Features
- **Automatic Scraping**: Scrape torrent data across a range of pages as specified.
- **Rate Limiter**: Avoid being banned with a configurable rate limiter.
- **Full-Text Search**: Perform searches across page titles, torrent names, and file paths inside torrent files.
- **Blazing fast**: Utilizes SQLite FTS5 for rapid searches (milliseconds response for over 250k torrents and 25m paths).
- **Advanced Search**: Use multi-processing for regular expression searching. Slower but more powerful and comprehensive.

## Installation

To get started, you'll need Python 3.10+. Then clone the GitHub repository and install the required packages:

```bash
git clone https://github.com/libertypi/mtsearch.git

cd mtsearch
pip install -r requirements.txt
```

## Usage

1. Upon the first run, a default configuration file (`config.json`) will be generated in the profile directory (default: `<script_dir>/profile`). Edit this config file before running this script again.
2. Scrape some data into the database: `mtsearch.py update`
3. Perform searches: `mtsearch.py search "your keyword"`

### Configuration File

The script uses a configuration file (`config.json`) with the following fields:

- `api_key`: Your API key.
- `domain`: The URL of the M-Team site. Leave this empty to use the default domain.
- `request_interval`: The time interval (in seconds) between each API request. Set to `0` to make requests without any delay.
- `hourly_limit`: The maximum number of requests permitted per hour. Set to `0` for no limit.
- `nord_user` / `nord_pass`: NordVPN service credentials for proxy rotation, used to bypass API throttling. These are not your account credentials.
- `search_params`: A list of parameters for the `/api/torrent/search` API used during the execution of `mtsearch.py update -p`. These parameters determine the scope and type of data retrieved. For a complete list of available parameters, consult the official M-Team API documentation.
  Common parameters include:
  - `mode`: Determines the type of content to search. Available values are `normal`, `adult`, `movie`, `music`, `tvshow`, `waterfall`, `rss`, `rankings`.
  - `categories`: An array of integers that identify specific categories within a selected mode. An empty array includes all categories.

Example `config.json`:

```json
{
    "api_key": "your_api_key_here",
    "domain": "",
    "request_interval": 10,
    "hourly_limit": 0,
    "nord_user": "",
    "nord_pass": "",
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
usage: mtsearch.py search [-h] [-P PROFILE] [-l | -f | -e] [pattern]

positional arguments:
  pattern               specify the search pattern

options:
  -h, --help            show this help message and exit
  -P, --profile PROFILE
                        profile directory (default: <script_dir>/profile)
  -l, --literal         use literal FTS5 matching (default)
  -f, --fts             use FTS5 matching (operators enabled)
  -e, --regex           use regular expression searching

examples:
  mtsearch.py search "foo"
  mtsearch.py search -f "foo OR bar"
  mtsearch.py search -e "202[2-4]"
```

- Enter interactive search mode (use -l, -f, -e to specify search modes):

  `mtsearch.py search` or `mtsearch.py s`

- Search using default literal FTS5 matching:

  `mtsearch.py search "foo"`

- Search using [FTS5](https://www.sqlite.org/fts5.html) syntax:

  `mtsearch.py search -f "foo OR bar"`

- Search using a regular expression (e.g., matches 2022, 2023, 2024):

  `mtsearch.py search -e "202[2-4]"`

#### For updating

```
usage: mtsearch.py update [-h] [-P PROFILE] [-d DUMP_DIR] [--no-limit] [-p PAGES | -i ID [ID ...] | --recreate]

options:
  -h, --help            show this help message and exit
  -P, --profile PROFILE
                        profile directory (default: <script_dir>/profile)
  -d DUMP_DIR           save torrent files to this directory
  --no-limit            temporarily disable rate limiting

actions:
  If no action is provided, defaults to: -p 3.

  -p PAGES              scrape one or more listing pages (e.g., '1-5' or '3')
  -i ID [ID ...]        update one or more torrent IDs
  --recreate            recreate the database

examples:
  mtsearch.py update -p 10-20
  mtsearch.py update -i 3 5 7
```

- Scrape the 5 most recent pages, bypassing the rate limiter.

  `mtsearch.py update -p 5 --no-limit`

- Scrape torrent ID 3, 5, and 7.

  `mtsearch.py update -i 3 5 7`

## Data File

A SQLite database named `data.db` will be created in the profile directory, storing all scraped torrent data. Ensure you back up this database as needed.

## API throttling:

> **Note**: Currently, M-Team implements dynamic and daily API throttling. You're free to experiment with different rate limitings until you get banned. For scraping a small number of pages, use the `--no-limit` switch to temporarily disable the rate limiter.

## Authors

- David Pi
