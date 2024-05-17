#!/usr/bin/env python3

"""
MTeam Scraper and Search Utility
================================

Description:
------------
This script is a powerful utility for scraping torrent data from M-Team torrent
sites and storing it in a local database for fast searching. It supports various
search modes including fixed-string matching, SQLite FTS5 matching, and regular
expression matching.

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

- `api_key`: API key for M-Team site.
- `domain`: The URL of the M-Team site. Leave empty to use the default domain.
- `hourly_limit`: Maximum number of requests in an hour. Set to 0 to disable.
- `request_interval`: Time interval between each request in seconds. Set to 0 to
  disable.
- `mode_categories`: List of modes and their subcategories to scrape. `mode` can
  be: `normal`, `adult`, `movie`, `music`, `tvshow`, `waterfall`, `rss`,
  `rankings`. `categories` is a list of integers where an empty list includes
  all subcategories.

Authors:
--------
- David Pi
- ChatGPT - OpenAI
- GitHub: https://github.com/libertypi/mtsearch
"""

import argparse
import dataclasses
import json
import logging
import re
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import attrgetter
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
import urllib3

logger = logging.getLogger(__name__)
stderr_write = sys.stderr.write
_DOMAIN = "https://kp.m-team.cc"


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """Data class to hold the result of a torrent search operation."""

    id: int
    title: str
    name: str
    date: int
    length: int
    files: list = dataclasses.field(default_factory=list)


class Database:
    """
    Database class for managing SQL operations related to torrents.
    """

    not_regex = frozenset(r"[]{}().*?+\|^$").isdisjoint

    def __init__(self, db_path: Path):
        """Initialize the database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

        # Create tables and triggers
        self.c.executescript(
            """
            BEGIN;

            -- Create Main Tables
            CREATE TABLE IF NOT EXISTS torrents(
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                name TEXT NOT NULL,
                date INTEGER NOT NULL,
                length INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files(
                id INTEGER NOT NULL,
                path TEXT NOT NULL,
                length INTEGER NOT NULL,
                FOREIGN KEY(id) REFERENCES torrents(id)
            );

            -- Create FTS Tables
            CREATE VIRTUAL TABLE IF NOT EXISTS torrents_fts USING fts5(
                title,
                name,
                date UNINDEXED,
                length UNINDEXED,
                content='torrents',
                content_rowid='id'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                id UNINDEXED,
                path,
                length UNINDEXED,
                content='files',
                content_rowid='rowid'
            );

            -- Create Indexes
            CREATE INDEX IF NOT EXISTS idx_files_id ON files(id);

            -- Create Triggers for FTS
            CREATE TRIGGER IF NOT EXISTS insert_torrents_fts AFTER INSERT ON torrents
            BEGIN
                INSERT INTO torrents_fts(rowid, title, name, date, length) VALUES (new.id, new.title, new.name, new.date, new.length);
            END;
            CREATE TRIGGER IF NOT EXISTS delete_torrents_fts AFTER DELETE ON torrents
            BEGIN
                INSERT INTO torrents_fts(torrents_fts, rowid, title, name, date, length) VALUES('delete', old.id, old.title, old.name, old.date, old.length);
            END;
            CREATE TRIGGER IF NOT EXISTS update_torrents_fts AFTER UPDATE ON torrents
            BEGIN
                INSERT INTO torrents_fts(torrents_fts, rowid, title, name, date, length) VALUES('delete', old.id, old.title, old.name, old.date, old.length);
                INSERT INTO torrents_fts(rowid, title, name, date, length) VALUES (new.id, new.title, new.name, new.date, new.length);
            END;
            CREATE TRIGGER IF NOT EXISTS insert_files_fts AFTER INSERT ON files
            BEGIN
                INSERT INTO files_fts(rowid, id, path, length) VALUES (new.rowid, new.id, new.path, new.length);
            END;
            CREATE TRIGGER IF NOT EXISTS delete_files_fts AFTER DELETE ON files
            BEGIN
                INSERT INTO files_fts(files_fts, rowid, id, path, length) VALUES('delete', old.rowid, old.id, old.path, old.length);
            END;
            CREATE TRIGGER IF NOT EXISTS update_files_fts AFTER UPDATE ON files
            BEGIN
                INSERT INTO files_fts(files_fts, rowid, id, path, length) VALUES('delete', old.rowid, old.id, old.path, old.length);
                INSERT INTO files_fts(rowid, id, path, length) VALUES (new.rowid, new.id, new.path, new.length);
            END;

            COMMIT;
            """
        )

    def __enter__(self):
        """Enable use of 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database connection when exiting the 'with' block."""
        self.close()

    def close(self):
        """Close the database connection."""
        self.c.execute("PRAGMA optimize")
        self.conn.close()

    def contains_id(self, tid: int):
        """Check if a torrent ID exists in the database."""
        return bool(
            self.c.execute(
                "SELECT EXISTS(SELECT 1 FROM torrents WHERE id = ?)", (tid,)
            ).fetchone()[0]
        )

    def get_total(self) -> int:
        """Get the total number of rows in the 'torrents' table."""
        return self.c.execute("SELECT COUNT(*) FROM torrents").fetchone()[0]

    def insert_torrent(self, detail: tuple, files: tuple):
        """Insert the metadata of a new torrent into the database."""
        with self.conn:
            self.c.execute(
                "INSERT INTO torrents (id, title, name, date, length) VALUES (?, ?, ?, ?, ?)",
                detail,
            )
            self.c.executemany(
                "INSERT INTO files (id, path, length) VALUES (?, ?, ?)",
                files,
            )

    def search(
        self, pattern: str, search_mode: str = "fixed"
    ) -> Iterable[SearchResult]:
        """
        Perform a search on the database based on the given pattern and search mode.

        Args:
            pattern: The search pattern.
            search_mode: The search mode, either 'fixed', 'fts', or 'regex'.

        Returns:
            An iterable of SearchResult objects.
        """

        # Full-Text Search (FTS) based search
        if search_mode in ("fixed", "fts"):
            return self._common_search(
                q1="SELECT rowid, title, name, date, length FROM torrents_fts WHERE title MATCH :pat OR name MATCH :pat",
                q2="""SELECT torrents.id, torrents.title, torrents.name, torrents.date, torrents.length, files_fts.path, files_fts.length
                      FROM files_fts
                      JOIN torrents ON files_fts.id = torrents.id
                      WHERE files_fts.path MATCH :pat""",
                param={"pat": f'"{pattern}"' if search_mode == "fixed" else pattern},
                c=self.c,
            ).values()

        # Regular expression search
        if search_mode == "regex":
            if self.not_regex(pattern):
                m = re.search(r"[^\W_]+", pattern)
                if not m:
                    pass
                # If pattern is a simple word, use LIKE matching
                elif m[0] == pattern:
                    return self._common_search(
                        q1="SELECT id, title, name, date, length FROM torrents WHERE title LIKE :pat OR name LIKE :pat",
                        q2="""SELECT torrents.id, torrents.title, torrents.name, torrents.date, torrents.length, files.path, files.length
                              FROM files
                              JOIN torrents ON files.id = torrents.id
                              WHERE files.path LIKE :pat""",
                        param={"pat": f"%{pattern}%"},
                        c=self.c,
                    ).values()
                # Convert pattern to a fuzzy regex
                else:
                    pattern = re.sub(r"[\W_]+", r"[\\W_]*", pattern)
            return self._regex_search(pattern).values()

        raise ValueError(f"Invalid search mode: {search_mode}")

    def _regex_search(self, pattern: str):
        """Perform a regular expression search using multi-processing."""

        result = {}
        futures = []
        query = "SELECT id FROM torrents ORDER BY id LIMIT 1 OFFSET ?"
        args = (
            self._re_worker,
            self.db_path.as_uri() + "?mode=ro",  # read-only
            re.compile(pattern, re.IGNORECASE).search,
        )

        with ProcessPoolExecutor() as ex:
            # Note: Using `ex._max_workers` to get the number of workers.
            per_worker, remainder = divmod(self.get_total(), ex._max_workers)

            # Query boundary IDs and distribute tasks among workers
            row = 0
            start_id = self.c.execute(query, (0,)).fetchone()[0]
            for _ in range(ex._max_workers):
                row += per_worker
                if remainder > 0:
                    row += 1
                    remainder -= 1
                end_id = self.c.execute(query, (row - 1,)).fetchone()[0]
                futures.append(ex.submit(*args, start_id, end_id))
                start_id = end_id

            # Collect multi-processing results
            futures = as_completed(futures)
            for future in futures:
                result.update(future.result())

        return result

    @staticmethod
    def _re_worker(uri: str, searcher, start_id: int, end_id: int):
        """
        Worker function to perform regex search on a chunk of data in a
        multi-processing environment.

        Args:
            uri: The uri path to the SQLite database.
            searcher: The search method of a compiled regex pattern.
            start_id: The starting ID for the range of rows this worker will handle.
            end_id: The ending ID for the range of rows this worker will handle.
        """

        # Establish a new database connection for this worker
        conn = sqlite3.connect(uri, uri=True)
        conn.create_function(
            "RESEARCH", 1, lambda s: searcher(s) is not None, deterministic=True
        )

        # Perform the regex search using the common search method and close the connection.
        result = Database._common_search(
            q1="""SELECT id, title, name, date, length FROM torrents
                WHERE id BETWEEN ? AND ?
                AND (RESEARCH(title) OR RESEARCH(name))""",
            q2="""SELECT torrents.id, torrents.title, torrents.name, torrents.date, torrents.length, files.path, files.length
                FROM files
                JOIN torrents ON files.id = torrents.id
                WHERE files.id BETWEEN ? AND ? AND RESEARCH(files.path)""",
            param=(start_id, end_id),
            c=conn.cursor(),
        )
        conn.close()

        return result

    @staticmethod
    def _common_search(q1: str, q2: str, param, c: sqlite3.Cursor):
        """
        Execute common search queries for torrents and files.

        Args:
            q1: SQL query for torrents table.
            q2: SQL query for files table.
            param: Parameters for the SQL query.
            c: SQLite cursor object.

        Returns:
            A dictionary of SearchResult objects.
        """
        result = {}
        for tid, title, name, date, total in c.execute(q1, param):
            result[tid] = SearchResult(tid, title, name, date, total)
        for tid, title, name, date, total, path, length in c.execute(q2, param):
            v = result.get(tid)
            if v is None:
                v = result[tid] = SearchResult(tid, title, name, date, total)
            v.files.append((path, length))
        return result


class RateLimiter:
    """
    A rate limiter for controlling global and per-request frequencies. None or
    non-positive values disable the corresponding rate limit.

    Parameters:
     - hourly_limit: The maximum number of requests allowed in an hour for
       global rate limiting.
     - request_interval: The minimum time interval (in seconds) between
       consecutive requests for per-request rate limiting.
    """

    def __init__(self, hourly_limit: int = None, request_interval: int = None):
        self._waitlist = []
        self.request_que = deque()
        self.last_request = 0

        if hourly_limit and hourly_limit > 0:
            self.hourly_limit = hourly_limit
            self._waitlist.append(self.wait_global)
        else:
            self.hourly_limit = None

        if request_interval and request_interval > 0:
            self.request_interval = request_interval
            self._waitlist.append(self.wait_request)
        else:
            self.request_interval = None

    def wait_global(self):
        """Global rate limiting logic."""
        que = self.request_que
        oldest_allowed = time.time() - 3600
        while que and que[0] <= oldest_allowed:
            self.request_que.popleft()

        if len(que) >= self.hourly_limit:
            sleep = que[0] - oldest_allowed
            stderr_write(
                f"Global rate limit reached. Waiting for {sleep:.2f} seconds. Hourly limit: {self.hourly_limit}.\n"
            )
            time.sleep(sleep)

    def wait_request(self):
        """Per-request rate limiting logic."""
        sleep = self.request_interval - time.time() + self.last_request
        if sleep > 0:
            stderr_write(
                f"Waiting for {sleep:.2f}s. Request interval: {self.request_interval:.2f} seconds.\n"
            )
            time.sleep(sleep)

    def __enter__(self):
        """Invoke all active rate limiting checks."""
        for func in self._waitlist:
            func()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Record the time of the most recent request."""
        self.last_request = time.time()
        if self.hourly_limit:
            self.request_que.append(self.last_request)


class MTeamScraper:
    """A scraper for downloading torrent information from M-Team."""

    _pageSize = 100
    _sleep = 120

    def __init__(
        self,
        api_key: str,
        domain: str,
        database: Database,
        ratelimiter: RateLimiter,
    ):
        """Initializes the MTeamScraper instance."""
        if not api_key:
            raise ValueError("api_key is null.")
        self.api_key = api_key

        self._search_url = urljoin(domain, "/api/torrent/search")
        self._detail_url = urljoin(domain, "/api/torrent/detail")
        self._files_url = urljoin(domain, "/api/torrent/files")

        self.database = database
        self.ratelimiter = ratelimiter
        self._init_session()

    def _init_session(self):
        """Initialize a requests session with retries and headers."""
        self.session = s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.Retry(
                total=7,
                status_forcelist={429, 500, 502, 503, 504, 521, 524},
                backoff_factor=0.5,
            )
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update(
            {
                "x-api-key": self.api_key,
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.",
            }
        )

    def _request_json(self, url: str, **kwargs) -> dict:
        """Make a POST request and handle rate limiting and retries."""
        while True:
            with self.ratelimiter:
                stderr_write(f"Connecting: {url}\n")
                for k in "json", "data":
                    if k in kwargs:
                        stderr_write(f"  {kwargs[k]}\n")
                res = self.session.post(url=url, **kwargs)
            res.raise_for_status()
            res = res.json()

            message = res["message"]
            if "SUCCESS" in message.upper():
                return res
            elif "請求過於頻繁" in message:
                stderr_write(f"Throttled. Waiting for {self._sleep} seconds.\n")
                time.sleep(self._sleep)
            elif "key無效" in message:
                raise Exception("Invalid API key. Check credential in config file.")
            else:
                raise Exception(message)

    def _get_list(self, mode: str, categories: list, page: int) -> list:
        """Get a list of torrents from the M-Team API."""
        return self._request_json(
            url=self._search_url,
            headers={"Content-Type": "application/json"},
            json={
                "mode": mode,
                "categories": categories or (),
                "pageNumber": page + 1,
                "pageSize": self._pageSize,
                "visible": 0,
            },
        )["data"]["data"]

    def _get_detail(self, tid: int):
        """Get details of a specific torrent."""
        return self._request_json(
            url=self._detail_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"id": tid},
        )["data"]

    def _get_files(self, tid: int):
        """Get files of a specific torrent."""
        return self._request_json(
            url=self._files_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"id": tid},
        )["data"]

    def scrape(self, mode: str, categories: list, page_range: range):
        """Scrape torrents from the M-Team site and store them in the database."""
        contains_id = self.database.contains_id
        removesuffix = re.compile(r"\.torrent$", re.I).sub

        for page in page_range:
            for item in self._get_list(mode, categories, page):
                tid = item.get("id")
                try:
                    tid = int(tid)
                    if contains_id(tid):
                        continue
                    detail = self._get_detail(tid)

                    title = detail["name"]
                    if not (isinstance(title, str) and title):
                        raise ValueError(f"Invalid title '{title}'.")

                    name = removesuffix("", detail["originFileName"], 1)
                    if not name:
                        raise ValueError("Empty torrent name.")

                    date = strptime(detail["createdDate"])
                    if not date >= 0:
                        logger.warning(f"Invalid creation date: '{date}'.")

                    if get_int(detail, "numfiles") > 1:
                        files = tuple(
                            (tid, f["name"], get_int(f, "size"))
                            for f in self._get_files(tid)
                        )
                        if not all(f[1] and f[2] >= 0 for f in files):
                            logger.warning(f"Invalid file path.")
                        length = sum(f[2] for f in files)
                    else:
                        files = ()
                        length = get_int(detail, "size")

                    if not length >= 0:
                        logger.warning(f"Invalid torrent length: '{length}'.")

                    self.database.insert_torrent(
                        detail=(tid, title, name, date, length),
                        files=files,
                    )
                except Exception as e:
                    logger.error(f"ID: '{tid}'. {e}")


def get_int(d: dict, k) -> int:
    """Returns the integer value for key k in dict d, or 0 if invalid."""
    try:
        return int(d[k])
    except (KeyError, TypeError, ValueError):
        return 0


def strptime(string: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> int:
    """Convert a date string to epoch time."""
    try:
        return int(time.mktime(time.strptime(string, fmt)))
    except (TypeError, ValueError):
        return 0


def strftime(epoch: int, fmt: str = "%F") -> str:
    """Convert epoch time to a formatted date string."""
    if epoch is None:
        raise TypeError("Epoch argument cannot be None.")
    return time.strftime(fmt, time.gmtime(epoch))


def humansize(size: int) -> str:
    """Convert bytes to human readable sizes."""
    for suffix in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if size < 1024:
            return f"{size:.2f} {suffix}B"
        size /= 1024
    return f"{size:.2f} YiB"


def config_logger(logger: logging.Logger, logfile: Path = None):
    """Configure logger to log to console and optionally to a file."""
    logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    if logfile:
        handler = logging.FileHandler(logfile)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M",
            )
        )
        logger.addHandler(handler)


def parse_args():
    def parse_range(range_str: str):
        m = re.fullmatch(r"(?:(\d+)-)?(\d+)", range_str)
        if m:
            return range(int(m[1] or 0), int(m[2]))
        raise argparse.ArgumentTypeError(
            "Invalid range format. Use 'start-end' or 'end'."
        )

    parser = argparse.ArgumentParser(
        description="""\
A Powerful MTeam Scraper and Search Utility.

Scrapes torrent data from M-Team torrent sites and stores it in a local SQLite
database. Supports various search modes including fixed-string matching, SQLite
FTS5 matching, and regular expression matching.

Author: David Pi
Contributor: ChatGPT - OpenAI""",
        epilog="""\
Examples:
  %(prog)s -m "girl OR woman"
  %(prog)s -u -r 5-10
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    exgroup = parser.add_mutually_exclusive_group()
    exgroup.add_argument(
        "-s",
        "--search",
        dest="mode",
        action="store_const",
        const="search",
        help="Search for a pattern in the database (default)",
    )
    exgroup.add_argument(
        "-u",
        "--update",
        dest="mode",
        action="store_const",
        const="update",
        help="Update the database",
    )
    exgroup.set_defaults(mode="search")

    group = parser.add_argument_group("Search Options")
    exgroup = group.add_mutually_exclusive_group()
    exgroup.add_argument(
        "-e",
        "--regex",
        dest="search_mode",
        action="store_const",
        const="regex",
        help="Use regular expression matching",
    )
    exgroup.add_argument(
        "-f",
        "--fixed",
        dest="search_mode",
        action="store_const",
        const="fixed",
        help="Use fixed-string matching (default)",
    )
    exgroup.add_argument(
        "-m",
        "--fts",
        dest="search_mode",
        action="store_const",
        const="fts",
        help="Use freeform SQLite FTS5 matching",
    )
    exgroup.set_defaults(search_mode="fixed")
    group.add_argument(
        "pattern",
        action="store",
        nargs="?",
        help="Specify the search pattern",
    )

    group = parser.add_argument_group("Update options")
    group.add_argument(
        "-r",
        dest="page_range",
        type=parse_range,
        default="3",
        help="Specify page range as 'start-end', or 'end' (Default: %(default)s)",
    )
    group.add_argument(
        "--no-limit",
        dest="nolimit",
        action="store_true",
        help="Disable rate limit defined in the config file",
    )

    return parser.parse_args()


def parse_config(config_path: Path) -> dict:
    """Parse the JSON-formatted configuration file located at `config_path`."""
    config = {
        "api_key": "",
        "domain": _DOMAIN,
        "hourly_limit": 100,
        "request_interval": 10,
        "mode_categories": [{"mode": "adult", "categories": []}],
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
        return config
    except FileNotFoundError:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        stderr_write(
            f"A blank configuration file has been created at '{config_path}'. "
            "Edit the settings before running this script again.\n"
        )
    except json.JSONDecodeError:
        stderr_write(
            f"The configuration file at '{config_path}' is malformed. "
            "Please fix the JSON syntax.\n"
        )
    sys.exit(1)


def _search(pattern: str, db: Database, search_mode: str, domain: str):
    """Perform a search in the local database and print the results."""
    sec = time.time()
    results = db.search(pattern, search_mode)
    sec = time.time() - sec

    # Sort by date
    results = sorted(results, key=attrgetter("date"))

    # Print search results
    sep = "=" * 80 + "\n"
    f1 = "{:>5}: {}\n".format
    f2 = "{:>5}: {}  [{}]\n".format
    f3 = "{:>5}  {}  [{}]\n".format
    detail_url = urljoin(domain, "detail")
    write = sys.stdout.write

    for s in results:
        write(sep)
        write(f1("Title", s.title))
        write(f1("Name", s.name))
        write(f1("Date", strftime(s.date)))
        write(f1("Size", humansize(s.length)))
        write(f1("ID", s.id))
        write(f1("URL", f"{detail_url}/{s.id}"))
        if s.files:
            files = iter(s.files)
            p, l = next(files)
            write(f2("Files", p, humansize(l)))
            for p, l in files:
                write(f3("", p, humansize(l)))

    write(
        f"{sep}Found {len(results):,} results in {db.get_total():,} torrents ({sec:.4f} seconds).\n"
    )


def main():
    args = parse_args()
    join_root = Path(__file__).parent.joinpath
    conf = parse_config(join_root("config.json"))
    db = Database(join_root("data.db"))
    try:
        domain = conf["domain"] or _DOMAIN
        if args.mode == "search":
            config_logger(logger)
            param = (db, args.search_mode, domain)
            if args.pattern:
                _search(args.pattern, *param)
            else:
                while True:
                    # Strip white-space only in interactive mode
                    pattern = input("Search pattern: ").strip()
                    if not pattern:
                        break
                    _search(pattern, *param)
        elif args.mode == "update":
            config_logger(logger, join_root("logfile.log"))
            if args.nolimit:
                limiter = RateLimiter()
            else:
                limiter = RateLimiter(conf["hourly_limit"], conf["request_interval"])
            mt = MTeamScraper(
                api_key=conf["api_key"],
                domain=domain,
                database=db,
                ratelimiter=limiter,
            )
            for m in conf["mode_categories"]:
                mt.scrape(
                    mode=m["mode"],
                    categories=m["categories"],
                    page_range=args.page_range,
                )
        else:
            raise ValueError(f"Invalid operation mode: {args.mode}")
    except Exception as e:
        logger.error(e)
    finally:
        db.close()


if __name__ == "__main__":
    main()
