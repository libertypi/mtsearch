#!/usr/bin/env python3

"""
MTeam Scraper and Search Utility
================================

Description:
------------
This script is a utility for scraping torrent data from M-Team torrent sites and
storing it in a local database for fast searching. It supports various search
modes including SQLite FTS5 matching and regular expression matching.

Main Functionality:
-------------------
1. Search for torrents in the local database.
2. Update the local database by scraping new torrents from the website.

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
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import attrgetter
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urljoin

import requests
import urllib3
from bencoder import bdecode  # bencoder.pyx

_DOMAIN = "https://api.m-team.cc"
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Torrent:
    """Data class to hold the details of a torrent."""

    id: int
    category: int
    title: str
    name: str
    date: int
    length: int
    files: list = dataclasses.field(default_factory=list)


class Database:
    """Database class for managing SQL operations related to torrents."""

    _SCHEMA = """
    BEGIN;

    -- Create Main Tables
    CREATE TABLE IF NOT EXISTS torrents(
        id INTEGER PRIMARY KEY,
        category INTEGER NOT NULL,
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
    CREATE TABLE IF NOT EXISTS categories(
        id INTEGER PRIMARY KEY,
        parent INTEGER,
        nameChs TEXT,
        nameCht TEXT,
        nameEng TEXT
    );

    -- Create FTS Tables
    CREATE VIRTUAL TABLE IF NOT EXISTS torrents_fts USING fts5(
        category UNINDEXED,
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
        INSERT INTO torrents_fts(rowid, category, title, name, date, length) VALUES (new.id, new.category, new.title, new.name, new.date, new.length);
    END;
    CREATE TRIGGER IF NOT EXISTS delete_torrents_fts AFTER DELETE ON torrents
    BEGIN
        INSERT INTO torrents_fts(torrents_fts, rowid, category, title, name, date, length) VALUES('delete', old.id, old.category, old.title, old.name, old.date, old.length);
    END;
    CREATE TRIGGER IF NOT EXISTS update_torrents_fts AFTER UPDATE ON torrents
    BEGIN
        INSERT INTO torrents_fts(torrents_fts, rowid, category, title, name, date, length) VALUES('delete', old.id, old.category, old.title, old.name, old.date, old.length);
        INSERT INTO torrents_fts(rowid, category, title, name, date, length) VALUES (new.id, new.category, new.title, new.name, new.date, new.length);
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
    _INS_TORRENTS = "INSERT INTO torrents (id, category, title, name, date, length) VALUES (?, ?, ?, ?, ?, ?)"
    _INS_FILES = "INSERT INTO files (id, path, length) VALUES (?, ?, ?)"
    _UPD_CATEGORIES = """
    INSERT INTO categories (id, parent, nameChs, nameCht, nameEng)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        parent = excluded.parent,
        nameChs = excluded.nameChs,
        nameCht = excluded.nameCht,
        nameEng = excluded.nameEng
    WHERE
        categories.parent != excluded.parent OR
        categories.nameChs != excluded.nameChs OR
        categories.nameCht != excluded.nameCht OR
        categories.nameEng != excluded.nameEng
    """

    def __init__(self, path):
        """Initialize the database."""
        self.path = path if isinstance(path, Path) else Path(path)
        self.conn = sqlite3.connect(self.path)
        self.c = self.conn.cursor()

        # Create tables and triggers
        self.c.executescript(self._SCHEMA)

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

    def update_categories(self, categories: tuple):
        """Update categories when there is a real change."""
        with self.conn:
            self.c.executemany(self._UPD_CATEGORIES, categories)

    def get_categories(self, column: str = "nameCht") -> dict:
        """Retrieve a dictionary of category IDs to the corresponding names."""
        return dict(self.c.execute(f"SELECT id, {column} FROM categories"))

    def get_total(self) -> int:
        """Get the total number of rows in the 'torrents' table."""
        return self.c.execute("SELECT COUNT(*) FROM torrents").fetchone()[0]

    def torrent_exists(self, tid: int) -> bool:
        """Check if a torrent ID exists in the database."""
        return (
            self.c.execute("SELECT 1 FROM torrents WHERE id = ?", (tid,)).fetchone()
            is not None
        )

    def torrent_up2date(self, tid: int, category: int, title: str) -> bool:
        """
        True if the torrent exists, False otherwise. Updates the torrent if the
        category or title has changed.
        """
        t = self.c.execute(
            "SELECT category, title FROM torrents WHERE id = ?", (tid,)
        ).fetchone()
        if not t:
            return False
        if t[0] != category or t[1] != title:
            logger.info(
                "Updating %s. Category: '%s' -> '%s'. Title: '%s' -> %s.", tid,
                t[0], category, t[1], title)  # fmt: skip
            with self.conn:
                self.c.execute(
                    "UPDATE torrents SET category = ?, title = ? WHERE id = ?",
                    (category, title, tid),
                )
        return True

    def insert_torrent(self, torrent: tuple, files: tuple):
        """Insert the metadata of a new torrent into the database."""
        with self.conn:
            self.c.execute(self._INS_TORRENTS, torrent)
            self.c.executemany(self._INS_FILES, files)

    def delete_torrent(self, tid: int):
        """Delete the torrent from the database."""
        with self.conn:
            self.c.execute("DELETE FROM torrents WHERE id = ?", (tid,))
            self.c.execute("DELETE FROM files WHERE id = ?", (tid,))

    def recreate(self):
        """Create a new database and insert the current data."""
        dest_path = self.path.with_stem(self.path.stem + "_new")
        if dest_path.exists():
            raise Exception(f"Destination exists: {dest_path}")

        logger.info("Creating new database: %s", dest_path)
        new_conn = sqlite3.connect(dest_path)
        new_c = new_conn.cursor()

        new_c.executescript(self._SCHEMA)
        with new_conn:
            new_c.executemany(
                self._INS_TORRENTS,
                self.c.execute("SELECT * from torrents ORDER BY id"),
            )
            new_c.executemany(
                self._INS_FILES,
                self.c.execute("SELECT * from files ORDER BY id, rowid"),
            )
            new_c.executemany(
                self._UPD_CATEGORIES,
                self.c.execute("SELECT * from categories ORDER BY id"),
            )

        logger.info("Data copying completed. Optimizing...")
        new_c.executescript("VACUUM; PRAGMA optimize; ANALYZE;")
        logger.info("Database optimization completed.")
        new_conn.close()


class Searcher:

    not_regex = frozenset(r"[]{}().*?+\|^$").isdisjoint

    def __init__(self, db: Database, domain: str = None) -> None:
        self.db = db
        self.detail_url = urljoin(domain or _DOMAIN, "detail/")
        self._categories = None

    @property
    def categories(self):
        if self._categories is None:
            self._categories = self.db.get_categories()
        return self._categories

    def search_print(self, pattern: str, mode: str):
        """Perform a search in the database and print the results."""
        sec = time.time()
        result = self.search(pattern, mode)
        sec = time.time() - sec

        # Sort by id
        result.sort(key=attrgetter("id"))

        # Print search results
        sep = "=" * 80 + "\n"
        f1 = "{:>5}: {}\n".format
        f2 = "{:>5}: {}  [{}]\n".format
        f3 = "{:>5}  {}  [{}]\n".format
        write = sys.stdout.write
        cat_get = self.categories.get

        for t in result:
            write(sep)
            write(f1("ID", t.id))
            write(f1("Cat", cat_get(t.category, "Unknown")))
            write(f1("Title", t.title))
            write(f1("Name", t.name))
            write(f1("Date", strftime(t.date)))
            write(f1("Size", humansize(t.length)))
            write(f1("URL", f"{self.detail_url}{t.id}"))
            if t.files:
                files = iter(t.files)
                p, l = next(files)
                write(f2("Files", p, humansize(l)))
                for p, l in files:
                    write(f3("", p, humansize(l)))

        write(
            f"{sep}Found {len(result):,} results in {self.db.get_total():,} torrents ({sec:.4f} seconds).\n"
        )

    def search(self, pattern: str, mode: str) -> List[Torrent]:
        """
        Perform a search on the database based on the given pattern and search mode.

        Args:
            pattern: The search pattern.
            search_mode: The search mode, either 'fixed', 'fts', or 'regex'.

        Returns:
            A list of Torrent objects.
        """

        # Full-Text Search (FTS) based search
        if mode in ("fixed", "fts"):
            result = self._common_search(
                q1="""
                SELECT rowid, *
                FROM torrents_fts
                WHERE title MATCH :pat OR name MATCH :pat
                """,
                q2="""
                SELECT files_fts.path, files_fts.length, torrents.*
                FROM files_fts
                JOIN torrents ON files_fts.id = torrents.id
                WHERE files_fts.path MATCH :pat
                """,
                param={"pat": f'"{pattern}"' if mode == "fixed" else pattern},
                c=self.db.c,
            )

        # Regular expression search
        elif mode == "regex":
            # For patterns that do not contain any regex metacharacters, while
            # containing word characters, convert to a fuzzy regex.
            if self.not_regex(pattern) and re.search(r"[^\W_]", pattern):
                pattern = re.sub(r"[\W_]+", r"[\\W_]*", pattern)

            result = self._regex_search(pattern)

        else:
            raise ValueError(f"Invalid search mode: {mode}")

        return list(result.values())

    def _regex_search(self, pattern: str):
        """Perform a regular expression search using multi-processing."""

        result = {}
        futures = []
        query = "SELECT id FROM torrents ORDER BY id LIMIT 1 OFFSET ?"
        db = self.db
        args = (
            self._re_worker,
            db.path.as_uri() + "?mode=ro",  # read-only
            re.compile(pattern, re.IGNORECASE).search,
        )

        with ProcessPoolExecutor() as ex:
            # Note: Using `ex._max_workers` to get the number of workers.
            per_worker, remainder = divmod(db.get_total(), ex._max_workers)

            # Query boundary IDs and distribute tasks among workers
            row = 0
            start_id = db.c.execute(query, (0,)).fetchone()[0]
            for _ in range(ex._max_workers):
                row += per_worker
                if remainder > 0:
                    row += 1
                    remainder -= 1
                end_id = db.c.execute(query, (row - 1,)).fetchone()[0]
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
        result = Searcher._common_search(
            q1="""
            SELECT * FROM torrents
            WHERE id BETWEEN ? AND ?
            AND (RESEARCH(title) OR RESEARCH(name))
            """,
            q2="""
            SELECT files.path, files.length, torrents.*
            FROM files
            JOIN torrents ON files.id = torrents.id
            WHERE torrents.id BETWEEN ? AND ? AND RESEARCH(files.path)
            """,
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
            A dictionary of Torrent objects.
        """
        result = {}
        for tid, *torrent in c.execute(q1, param):
            result[tid] = Torrent(tid, *torrent)
        for path, length, tid, *torrent in c.execute(q2, param):
            t = result.get(tid)
            if t is None:
                t = result[tid] = Torrent(tid, *torrent)
            t.files.append((path, length))
        return result


class RateLimiter:
    """
    A rate limiter for controlling global and per-request frequencies. None or
    non-positive values disable the corresponding rate limit.

    Parameters:
     - request_interval: The minimum time interval (in seconds) between
       consecutive requests for per-request rate limiting.
     - hourly_limit: The maximum number of requests allowed in an hour for
       global rate limiting.
    """

    def __init__(self, request_interval: int = None, hourly_limit: int = None):
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
            logger.info(
                f"Waiting for {sleep:.2f}s. Hourly limit: {self.hourly_limit}s."
            )
            time.sleep(sleep)

    def wait_request(self):
        """Per-request rate limiting logic."""
        sleep = self.request_interval - time.time() + self.last_request
        if sleep > 0:
            logger.info(
                f"Waiting for {sleep:.2f}s. Request interval: {self.request_interval:.2f}s."
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


class APIError(Exception):
    """Exception for API errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class CriticalAPIError(APIError):
    """Exception for critical API errors."""


class MTeamScraper:
    """A scraper for M-Team."""

    _PAGE_SIZE: int = 200
    _THROTTLE_TIMMER: int = 120

    def __init__(
        self,
        api_key: str,
        db: Database,
        ratelimiter: RateLimiter,
        domain: str = None,
        nordvpn_path: str = None,
        dump_dir: str = None,
    ):
        """Initialize the MTeamScraper instance."""
        if not api_key:
            raise ValueError("api_key is null.")
        self.api_key = api_key

        self._domain = domain or _DOMAIN
        self._url_cache = {}

        self.db = db
        self.ratelimiter = ratelimiter

        # Setup NordVPN
        self._nord_cmd = None
        if nordvpn_path:
            nordvpn_path = shutil.which(nordvpn_path)
            if nordvpn_path:
                self._nord_cmd = (
                    nordvpn_path,
                    "--connect" if os.name == "nt" else "connect",
                )
            else:
                logger.warning("NordVPN not found. VPN switching will be disabled.")

        # Setup cache directory
        self._dump_dir = None
        self._fetch_torrent = self._download_torrent
        if dump_dir:
            try:
                dump_dir = Path(dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error("Unable to access cache directory: %s", e)
            else:
                self._dump_dir = dump_dir
                self._fetch_torrent = self._dump_torrent

        self._init_session()

    def _init_session(self):
        """Initialize a requests session with retries and headers."""
        self.session = s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=urllib3.Retry(
                total=5,
                status_forcelist=frozenset((429, 500, 502, 503, 504, 521, 524)),
                backoff_factor=0.5,
            )
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update(
            {
                "x-api-key": self.api_key,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.",
            }
        )

    def update_categories(self):
        """Update categories in the database."""
        try:
            categories = self._request_api(
                path="/api/torrent/categoryList",
                ratelimit=False,
            )["data"]["list"]
            categories = tuple(
                (
                    int(d["id"]),
                    int(d["parent"]) if d["parent"] else None,
                    d["nameChs"],
                    d["nameCht"],
                    d["nameEng"],
                )
                for d in categories
            )
            self.db.update_categories(categories)
        except CriticalAPIError:
            raise
        except Exception as e:
            logger.error("Failed to update categories: %s", e)

    def from_search(self, params: dict, pages: Iterable[int]):
        """Update the database from the `search` API."""
        self._process_scrape(self._scrape_search(params, pages))

    def from_detail(self, tids: Iterable[int]):
        """Update the database from the `detail` API."""
        self._process_scrape(self._scrape_detail(tids))

    def _scrape_search(self, params: dict, pages: Iterable[int]):
        """Retrieve data from the `search` API."""
        params = dict(params)
        params.setdefault("pageSize", self._PAGE_SIZE)
        for p in pages:
            params["pageNumber"] = p
            yield from self._request_api(
                path="/api/torrent/search",
                ratelimit=False,
                headers={"Content-Type": "application/json"},
                json=params,
            )["data"]["data"]

    def _scrape_detail(self, tids: Iterable[int]):
        """Retrieve data from the `detail` API."""
        for tid in tids:
            try:
                yield self._request_api(
                    path="/api/torrent/detail",
                    data={"id": tid},
                )["data"]
            except CriticalAPIError:
                raise
            except APIError as e:
                if "種子未找到" in e.message and self.db.torrent_exists(tid):
                    logger.info("Remove torrent: %s. (%s)", tid, e.message)
                    self.db.delete_torrent(tid)
                else:
                    logger.error("ID: %s. %s", tid, e)
            except Exception as e:
                logger.error("ID: %s. %s", tid, e)

    def _process_scrape(self, items: Iterable[dict]):
        """Process and update the database with the provided scrape results."""
        torrent_up2date = self.db.torrent_up2date
        join = b"/".join

        for item in items:
            try:
                tid = int(item["id"])
                # title
                title = item["name"].strip()
                if not (isinstance(title, str) and title):
                    raise ValueError(f"Invalid title '{title}'.")
                # category
                category = get_int(item, "category")
                if category <= 0:
                    logger.error("ID: %s. Invalid category: '%s'.", tid, category)
                # check and update existing torrent
                if torrent_up2date(tid, category, title):
                    continue
                # download and decode the torrent
                data = bdecode(self._fetch_torrent(tid))
                # date
                date = get_int(data, b"creation date")
                if date <= 0:
                    date = strptime(item["createdDate"])
                    if date <= 0:
                        logger.error("ID: %s. Invalid creation date: '%s'.", tid, date)
                # name
                data = data[b"info"]
                k = b"name.utf-8" if b"name.utf-8" in data else b"name"
                name = data[k].decode(errors="ignore").strip()
                if not (isinstance(name, str) and name):
                    raise ValueError(f"Invalid name '{name}'.")
                # files & length
                files = data.get(b"files")
                if files:
                    k = b"path.utf-8" if b"path.utf-8" in files[0] else b"path"
                    files = tuple(
                        (tid, join(f[k]).decode(errors="ignore"), get_int(f, b"length"))
                        for f in files
                    )
                    if not all(f[1] and f[2] >= 0 for f in files):
                        logger.error("ID: %s. Invalid file path.", tid)
                    length = sum(f[2] for f in files)
                else:
                    files = ()
                    length = get_int(data, b"length")
                if length <= 0:
                    length = get_int(item, "size")
                    if length <= 0:
                        logger.error(
                            "ID: %s. Invalid torrent length: '%s'.", tid, length
                        )
                # insert into the database
                self.db.insert_torrent(
                    torrent=(tid, category, title, name, date, length),
                    files=files,
                )
            except CriticalAPIError:
                raise
            except Exception as e:
                logger.error("ID: %s. %s", item.get("id"), e)

    def _request_api(
        self,
        path: str,
        ratelimit: bool = True,
        headers: dict = {"Content-Type": "application/x-www-form-urlencoded"},
        **kwargs,
    ):
        """Make a POST request to the API and return JSON response."""
        try:
            url = self._url_cache[path]
        except KeyError:
            url = self._url_cache[path] = urljoin(self._domain, path)

        logger.info(
            "Requesting: %s. Payload: %s", url, kwargs.get("data") or kwargs.get("json")
        )
        while True:
            if ratelimit:
                with self.ratelimiter:
                    res = self.session.post(url, headers=headers, **kwargs)
            else:
                res = self.session.post(url, headers=headers, **kwargs)
            res.raise_for_status()
            res = res.json()

            message = res["message"]
            if "SUCCESS" in message.upper():
                return res
            self._handle_api_errs(message)

    def _handle_api_errs(self, message: str):
        """
        Handle API error messages. This method is to be used within a loop to
        manage throttling and retries.
        """
        if "請求過於頻繁" in message:
            if self._nord_cmd:
                logger.info("Throttled. Switching NordVPN. (%s)", message)
                subprocess.run(self._nord_cmd)
                self._init_session()
            else:
                logger.info(
                    "Throttled. Waiting for %ss. (%s)", self._THROTTLE_TIMMER, message
                )
                time.sleep(self._THROTTLE_TIMMER)
        elif "key無效" in message:
            raise CriticalAPIError(f"Invalid API key. ({message})")
        else:
            raise APIError(message)

    def _download_torrent(self, tid: int) -> bytes:
        """Download the torrent file for a given torrent ID."""
        url = self._request_api(
            path="/api/torrent/genDlToken",
            ratelimit=False,
            data={"id": tid},
        )["data"]

        logger.info("Downloading torrent: %s", tid)
        while True:
            with self.ratelimiter:
                res = self.session.get(url)
            res.raise_for_status()
            # JSON responses indicate error
            if "application/json" in res.headers.get("Content-Type", "").lower():
                message = res.json()["message"]
                if "下載配額用盡" in message:
                    raise CriticalAPIError(f"Download quota exhausted. ({message})")
                self._handle_api_errs(message)
            else:
                return res.content

    def _dump_torrent(self, tid: int):
        """Download and dump the torrent file."""
        file = self._dump_dir.joinpath(f"{tid}.torrent")
        if file.exists():
            logger.info("Using cache: %s", file.name)
            try:
                return file.read_bytes()
            except Exception as e:
                logger.warning(e)
        content = self._download_torrent(tid)
        try:
            file.write_bytes(content)
        except Exception as e:
            logger.warning(e)
            try:
                os.unlink(file)
            except Exception:
                pass
        return content


def get_int(d: dict, k) -> int:
    """Return the integer value for key k in dict d, or 0 if invalid."""
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


def config_logger(logger: logging.Logger = logger, logfile: Path = None):
    """Configure logger to log to console and optionally to a file."""
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

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
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)


def parse_args():
    def parse_range(string: str):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", string)
        if m:
            return range(int(m[1]), int(m[2] or m[1]) + 1)
        raise argparse.ArgumentTypeError(
            "Invalid range format. Use 'page' or 'start-end'."
        )

    parser = argparse.ArgumentParser(
        description="""\
A M-Team scraper and search utility.

Scrapes torrent data from M-Team website and stores it in a local database.
Supports SQLite FTS5 and regular expression matching.

Author: David Pi
Contributor: ChatGPT - OpenAI""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="mode", required=True)

    # search
    subparser = subparsers.add_parser(
        "search",
        aliases=("s",),
        help="search for patterns",
        epilog="""\
examples:
  %(prog)s "foo"
  %(prog)s -m "foo OR bar"
  %(prog)s -e "202[2-4]"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparser.set_defaults(mode="search")
    subparser.add_argument(
        "pattern",
        action="store",
        nargs="?",
        help="specify the search pattern",
    )
    exgroup = subparser.add_mutually_exclusive_group()
    exgroup.set_defaults(search_mode="fixed")
    exgroup.add_argument(
        "-e",
        "--regex",
        dest="search_mode",
        action="store_const",
        const="regex",
        help="use regular expression matching",
    )
    exgroup.add_argument(
        "-f",
        "--fixed",
        dest="search_mode",
        action="store_const",
        const="fixed",
        help="use fixed-string FTS5 matching (default)",
    )
    exgroup.add_argument(
        "-m",
        "--fts",
        dest="search_mode",
        action="store_const",
        const="fts",
        help="use freeform FTS5 matching",
    )

    # update
    subparser = subparsers.add_parser(
        "update",
        aliases=("u",),
        help="update the database",
        epilog="""\
examples:
  %(prog)s -p 1-10
  %(prog)s -i 3 5 7
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparser.set_defaults(mode="update")
    subparser.add_argument(
        "-d",
        dest="dump_dir",
        help="save torrent files to this directory",
    )
    subparser.add_argument(
        "--no-limit",
        dest="no_limit",
        action="store_true",
        help="temporarily disable rate limiting",
    )
    exgroup = subparser.add_argument_group("actions")
    exgroup = exgroup.add_mutually_exclusive_group(required=True)
    exgroup.add_argument(
        "-p",
        dest="pages",
        type=parse_range,
        const="1-3",
        nargs="?",
        help="scrape one or more pages (format: 'page' or 'start-end', default: %(const)s)",
    )
    exgroup.add_argument(
        "-i",
        dest="id",
        type=int,
        nargs="+",
        help="update one or more torrent IDs",
    )
    exgroup.add_argument(
        "--recreate",
        dest="recreate",
        action="store_true",
        help="recreate the database",
    )

    return parser.parse_args()


def parse_config(config_path: Path) -> dict:
    """Parse the JSON-formatted configuration file located at `config_path`."""
    config = {
        "api_key": "",
        "domain": _DOMAIN,
        "request_interval": 10,
        "hourly_limit": 100,
        "nordvpn_path": "",
        "search_params": [{"mode": "adult", "categories": []}],
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
        return config
    except FileNotFoundError:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        sys.exit(
            f"A blank configuration file has been created at '{config_path}'. "
            "Edit the settings before running this script again."
        )
    except json.JSONDecodeError:
        sys.exit(f"The configuration file at '{config_path}' is malformed.")


def main():

    args = parse_args()
    join_root = Path(__file__).parent.joinpath
    conf = parse_config(join_root("config.json"))

    db = Database(join_root("data.db"))
    try:
        if args.mode == "search":
            config_logger()

            searcher = Searcher(db, conf["domain"])

            if args.pattern:
                searcher.search_print(args.pattern, args.search_mode)
            else:
                while True:
                    pattern = input("Pattern: ").strip()
                    if not pattern:
                        break
                    searcher.search_print(pattern, args.search_mode)

        elif args.mode == "update":
            config_logger(logfile=join_root("logfile.log"))

            if args.recreate:
                db.recreate()
                return

            if args.no_limit:
                limiter = RateLimiter()
            else:
                limiter = RateLimiter(conf["request_interval"], conf["hourly_limit"])

            mt = MTeamScraper(
                api_key=conf["api_key"],
                db=db,
                ratelimiter=limiter,
                domain=conf["domain"],
                nordvpn_path=conf["nordvpn_path"],
                dump_dir=args.dump_dir,
            )
            mt.update_categories()

            if args.pages:
                for p in conf["search_params"]:
                    mt.from_search(params=p, pages=args.pages)
            elif args.id:
                mt.from_detail(tids=args.id)

    except Exception as e:
        logger.critical(e)

    finally:
        db.close()


if __name__ == "__main__":
    main()
