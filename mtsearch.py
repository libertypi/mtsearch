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

- `domain`: The URL of the M-Team site.
- `pages`: List of pages to scrape.
- `username`: Username for M-Team site (if "username" or "password" is null, prompt user at login).
- `password`: Password for M-Team site (optional).
- `maxRequests`: Maximum number of requests in a time window. Set to 0 or null to disable.
- `timeWindow`: Time window in seconds for maxRequests. Set to 0 or null to disable.
- `requestInterval`: Time interval between each request in seconds. Set to 0 or null to disable.

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
import pickle
import re
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from getpass import getpass
from operator import attrgetter
from pathlib import Path
from random import choice as random_choice
from typing import Iterable
from urllib.parse import urljoin

import requests
import urllib3
from bencoder import bdecode  # bencoder.pyx
from lxml.etree import XPath
from lxml.html import fromstring as html_fromstring

join_root = Path(__file__).parent.joinpath
stderr_write = sys.stderr.write


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """Data class to hold the result of a torrent search operation."""

    id: int
    title: str
    name: str
    date: int
    length: int
    files: list = dataclasses.field(default_factory=list)


class LogException(Exception):
    """This exception is intended to be logged when caught."""

    pass


class Database:
    """
    Database class for managing SQL operations related to torrents.
    """

    not_regex = frozenset(r"[]{}().*?+\|^$").isdisjoint

    def __init__(self, db_name: str = "data.db"):
        """Initialize the database."""
        self.db_path = join_root(db_name)
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

    def insert_torrent(self, tid: int, title: str, data: bytes):
        """Insert the metadata of a new torrent into the database."""

        # Decode the torrent
        data = bdecode(data)
        date = get_int(data, b"creation date")
        data = data[b"info"]
        name = (data[b"name.utf-8"] if b"name.utf-8" in data else data[b"name"]).decode(
            errors="ignore"
        )

        # Check if the torrent has multiple files
        files = data.get(b"files")
        if files:
            k = b"path.utf-8" if b"path.utf-8" in files[0] else b"path"
            join = b"/".join
            files = tuple(
                (tid, join(f[k]).decode(errors="ignore"), get_int(f, b"length"))
                for f in files
            )
            length = sum(f[2] for f in files)
        else:
            files = ()
            length = get_int(data, b"length")

        # Verification
        if not name:
            raise ValueError("Empty torrent name.")
        if not date >= 0:
            logging.warning(f"Invalid creation date: '{date}'. ID: {tid}")
        if not length >= 0:
            logging.warning(f"Invalid torrent length: '{length}'. ID: {tid}")
        if not all(f[1] and f[2] >= 0 for f in files):
            logging.warning(f"Invalid file path. ID: {tid}")

        # Insert into the database
        with self.conn:
            self.c.execute(
                "INSERT INTO torrents (id, title, name, date, length) VALUES (?, ?, ?, ?, ?)",
                (tid, title, name, date, length),
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
     - max_requests: The maximum number of requests allowed in a given time
       window for global rate limiting.
     - time_window: The time window (in seconds) for the global rate limiting.
     - request_interval: The minimum time interval (in seconds) between
       consecutive requests for per-request rate limiting.
    """

    def __init__(self, max_requests: int, time_window: int, request_interval: int):
        self._waitlist = []
        self.request_que = deque()
        self.last_request = 0

        # Configure global rate limiting
        if is_positive(max_requests) and is_positive(time_window):
            self.max_requests = max_requests
            self.time_window = time_window
            self._waitlist.append(self.wait_global)
        else:
            self.max_requests = float("inf")
            self.time_window = 0

        # Configure per-request rate limiting
        if is_positive(request_interval):
            self.request_interval = request_interval
            self._waitlist.append(self.wait_request)
        else:
            self.request_interval = 0

    def wait(self):
        """Invoke all active rate limiting checks."""
        for func in self._waitlist:
            func()

    def wait_global(self):
        """Global rate limiting logic."""
        que = self.request_que
        oldest_allowed = time.time() - self.time_window
        while que and que[0] <= oldest_allowed:
            self.request_que.popleft()

        if len(que) >= self.max_requests:
            sleep = que[0] - oldest_allowed
            stderr_write(
                f"Global rate limit reached. Waiting for {sleep:.2f} seconds. "
                f"Current setting: {self.max_requests} requests in {self.time_window} seconds\n"
            )
            time.sleep(sleep)

    def wait_request(self):
        """Per-request rate limiting logic."""
        sleep = self.request_interval - time.time() + self.last_request
        if sleep > 0:
            stderr_write(
                f"Waiting for {sleep:.2f}s. Current interval: {self.request_interval:.2f} seconds\n"
            )
            time.sleep(sleep)

    def record_request(self):
        """Record the time of the most recent request."""
        self.last_request = time.time()
        if self.time_window:
            self.request_que.append(self.last_request)

    def extend_interval(self, factor: float, max_interval: float):
        """Extend the request interval by a factor, up to a maximum value."""
        self.request_interval = min(self.request_interval * factor, max_interval)


class MTeamScraper:
    """A scraper for downloading torrent information from M-Team."""

    def __init__(
        self,
        domain: str,
        database: Database,
        cookies_name: str = "cookies",
        username: str = None,
        password: str = None,
        max_requests: int = None,
        time_window: int = None,
        request_interval: int = None,
        dump_dir: str = None,
    ):
        """Initializes the MTeamScraper instance."""
        self.domain = domain
        self.database = database
        self.credential = (
            {"username": username, "password": password}
            if username and password
            else None
        )
        self.ratelimiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            request_interval=request_interval,
        )

        # Setup dump dir if provided
        if dump_dir:
            self.dump_dir = Path(dump_dir)
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self._save_torrent = self._save_to_file_and_db
        else:
            self.dump_dir = None
            self._save_torrent = database.insert_torrent

        # Initialize cookie storage path and session
        self.cookie_path = join_root(cookies_name)
        self._init_session()

        # Initialize request method with login check
        self.get = self._login_and_get

        self.xp_throttle = xp_compile(
            "string(//h3/text()[re:test(., '请求.*?频繁')])",
            namespaces={"re": "http://exslt.org/regular-expressions"},
        )

    def _init_session(self):
        """Initialize a requests session with retries and user agent."""
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

        # Load user agents
        with open(join_root("useragents.txt"), "r", encoding="utf-8") as f:
            self.useragents = tuple(filter(None, map(str.strip, f)))
        if not self.useragents:
            raise ValueError("The user-agent list must not be empty.")
        s.headers["User-Agent"] = random_choice(self.useragents)

        try:
            with open(self.cookie_path, "rb") as f:
                s.cookies = pickle.load(f)
        except Exception:
            pass

    def dump_cookies(self):
        """Save session cookies to a file."""
        try:
            with open(self.cookie_path, "wb") as f:
                pickle.dump(self.session.cookies, f)
        except OSError as e:
            stderr_write(f"Unable to save cookies at '{self.cookie_path}': {e}\n")

    def _get(self, url: str, **kwargs):
        """Perform an HTTP GET request and return the response."""
        stderr_write(f"Connecting: {url} {kwargs or ''}\n")
        response = self.session.get(
            url=url,
            headers={"User-Agent": random_choice(self.useragents)},
            **kwargs,
        )
        response.raise_for_status()
        return response

    def _login_and_get(self, url: str, **kwargs):
        """Attempt to perform an HTTP GET request; log in if redirected"""
        response = self._get(url, **kwargs)
        if "/login.php" in response.url:
            while True:
                credential = self.credential or {
                    "username": input("Username: "),
                    "password": getpass("Password: "),
                }
                self.session.post(
                    url=urljoin(self.domain, "takelogin.php"),
                    data=credential,
                    headers={"referer": urljoin(self.domain, "login.php")},
                )
                response = self._get(url, **kwargs)
                if "/login.php" not in response.url:
                    self.dump_cookies()
                    break
                if self.credential:
                    raise LogException("Login failed. Check credential in config file.")
                stderr_write("Login failed. Try again.\n")
        self.get = self._get
        return response

    def get_tree(self, url: str, **kwargs):
        """Fetch a URL and parse it into an lxml HTML tree."""

        self.ratelimiter.wait()

        while True:
            # Make the request
            response = self.get(url, **kwargs)
            self.ratelimiter.record_request()

            # Parse the response
            tree = html_fromstring(response.content, base_url=response.url)

            # Check for throttling
            throttle_text = self.xp_throttle(tree)
            if not throttle_text:
                return tree

            try:
                # Extract sleep time from meta tag
                sleep = tree.xpath('number(//meta[@http-equiv="refresh"]/@content)')
                if not sleep > 0:
                    raise ValueError
                stderr_write(f"Throttled. Waiting for {sleep} seconds.\n")
            except ValueError:
                # Handle missing meta tag
                if "限制访问" in throttle_text:
                    raise LogException(
                        f"Critical Issue ('Fucked'): {throttle_text}. URL: {response.url}"
                    )
                sleep = 120  # default to 120s
                stderr_write(
                    f"Throttled and refresh tag not found. Waiting for {sleep} seconds. Message: {throttle_text}\n"
                )

            # Update minimum request interval
            self.ratelimiter.extend_interval(factor=1.5, max_interval=sleep)

            # Sleep for the required time
            time.sleep(sleep)

    def _get_pages(self, page: str, page_range: tuple):
        lo, hi = page_range
        if 0 < hi <= lo:
            raise ValueError("'end' must be greater than 'start' or 0 for no limit.")

        # Prepare the URL and fetch the first page
        url = urljoin(self.domain, page)
        params = {"incldead": 0, "page": lo}
        tree = self.get_tree(url, params=params)

        # Determine the total number of pages
        total = tree.xpath(
            'string(//td[@id="outer"]/table//td/p[@align="center"]/a[contains(@href, "page=")][last()]/@href)'
        )
        try:
            total = int(re.search(r"\bpage=(\d+)", total)[1]) + 1
        except Exception:
            raise LogException(f"Failed to identify pagination: '{tree.base_url}'")
        if not (0 < hi <= total):
            hi = total

        yield tree
        del tree
        for i in range(lo + 1, hi):
            params["page"] = i
            yield self.get_tree(url, params=params)

    def _get_title_from_details(self, tid: int):
        # "details.php" has an independent and more lenient rate control
        tree = html_fromstring(
            self.get(
                urljoin(self.domain, "details.php"),
                params={"id": tid},
            ).content
        )
        title = tree.xpath('string(//input[@name="torrent_name"]/@value)')
        return title or tree.xpath('string(.//h1[@id="top"]/text())')

    def scrape(self, page: str, page_range: tuple):
        """Scrape specified pages and update the database."""
        id_searcher = re.compile(r"\bid=(\d+)").search
        contains_id = self.database.contains_id

        # Prepare XPath queries for titles and links
        epath = './/form[@id="form_torrent"]/table[@class="torrents"]//table[@class="torrentname"]/tr[1]'
        xp_t1 = xp_compile('string(td[2]/a[contains(@href, "details.php?")]/@title)')
        xp_t2 = xp_compile('string(td[2]/a[contains(@href, "details.php?")])')
        xp_lk = xp_compile('string(td[3]/a[contains(@href, "download.php?")]/@href)')

        for tree in self._get_pages(page, page_range):
            for tree in tree.iterfind(epath):
                link = xp_lk(tree)
                try:
                    tid = int(id_searcher(link)[1])
                    if contains_id(tid):
                        continue
                    title = xp_t1(tree)
                    if not title:
                        title = xp_t2(tree)
                        if not title or title.endswith(".."):
                            title = self._get_title_from_details(tid)
                    title = title.strip()
                    if not title:
                        raise ValueError("Empty title.")
                    self._save_torrent(
                        tid, title, self.get(urljoin(self.domain, link)).content
                    )
                except Exception as e:
                    logging.error(f"Failed to process torrent at '{link}': {e}")

    def _save_to_file_and_db(self, tid: int, title: str, data: bytes):
        try:
            with open(self.dump_dir.joinpath(f"{tid}.torrent"), "wb") as f:
                f.write(data)
        except Exception as e:
            stderr_write(f"Saving torrent failed: {e}\n")
        self.database.insert_torrent(tid, title, data)


def is_positive(n) -> bool:
    return n is not None and n > 0


def get_int(d: dict, k) -> int:
    """Returns the integer value for key k in dict d, or 0 if invalid."""
    try:
        return int(d[k])
    except (KeyError, TypeError, ValueError):
        return 0


def xp_compile(path: str, **kwargs):
    return XPath(path, smart_strings=False, **kwargs)


def humansize(size: int) -> str:
    """Convert bytes to human readable sizes."""
    for suffix in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if size < 1024:
            return f"{size:.2f} {suffix}B"
        size /= 1024
    return f"{size:.2f} YiB"


def strftime(epoch: int, fmt: str = "%F") -> str:
    if epoch is None:
        raise TypeError("Epoch argument cannot be None.")
    return time.strftime(fmt, time.gmtime(epoch))


def config_logging(filename: str = "logfile.log"):
    """Send messages to both the log file and stderr."""
    # File handler
    logging.basicConfig(
        filename=join_root(filename),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(ch)


def parse_args():
    def parse_range(range_str: str):
        m = re.fullmatch(r"(?:(\d+)-)?(\d+)", range_str)
        if m:
            return int(m[1] or 0), int(m[2])
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
        help="Specify page range as 'start-end', 'end', or '0' for unlimited (Default: %(default)s)",
    )
    group.add_argument(
        "--dump",
        dest="dump_dir",
        help="Save torrent files to this directory",
    )
    group.add_argument(
        "--no-limit",
        dest="nolimit",
        action="store_true",
        help="Disable rate limit defined in the config file",
    )

    return parser.parse_args()


def parse_config(config_path) -> dict:
    """
    Parse the JSON-formatted configuration file located at `config_path`.

    Notes:
     - If "username" or "password" is null, prompt user at login.
    """
    config = {
        "domain": "https://xp.m-team.io",
        "pages": ["adult.php"],
        "username": None,
        "password": None,
        "maxRequests": 100,
        "timeWindow": 3600,
        "requestInterval": 20,
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
    sec = time.time()
    results = db.search(pattern, search_mode)
    sec = time.time() - sec

    # Sort by date
    results = sorted(results, key=attrgetter("date"))

    # Print search results
    sep = "=" * 80 + "\n"
    f1 = "{:6}: {}\n".format
    f2 = "{:6}: {}  [{}]\n".format
    f3 = "{:6}  {}  [{}]\n".format
    url = urljoin(domain, "details.php")
    write = sys.stdout.write

    for s in results:
        write(sep)
        write(f1("Title", s.title))
        write(f1("Name", s.name))
        write(f1("Date", strftime(s.date)))
        write(f1("Length", humansize(s.length)))
        write(f1("ID", s.id))
        write(f1("URL", f"{url}?id={s.id}"))
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
    config_logging()
    args = parse_args()
    conf = parse_config(join_root("config.json"))
    db = Database()
    try:
        if args.mode == "search":
            param = (db, args.search_mode, conf["domain"])
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
            if args.nolimit:
                conf["maxRequests"] = conf["requestInterval"] = None
            mt = MTeamScraper(
                domain=conf["domain"],
                database=db,
                username=conf["username"],
                password=conf["password"],
                max_requests=conf["maxRequests"],
                time_window=conf["timeWindow"],
                request_interval=conf["requestInterval"],
                dump_dir=args.dump_dir,
            )
            for page in conf["pages"]:
                mt.scrape(page, args.page_range)
        else:
            raise ValueError(f"Invalid operation mode: {args.mode}")
    except LogException as e:
        logging.error(e)
    except Exception as e:
        stderr_write(f"ERROR: {e}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
