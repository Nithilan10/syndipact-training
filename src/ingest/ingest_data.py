from abc import ABC, abstractmethod
import pandas as pd
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

try:
    import matplotlib.pyplot as plt
except Exception:
    plt=None

RE_URL = re.compile(r"http[s]?://\S+")
RE_MULTI_WS = re.compile(r"\s+")
RE_NONPRINT = re.compile(r"[\x00-\x1f\x7f-\x9f]")
RE_CODEBLOCK = re.compile(r"```.*?```", flags=re.DOTALL)
RE_INLINE_CODE = re.compile(r"`[^`]+`")

COMMITMENT_PATTERNS = [
    r"\b(i will|i'll|ill)\b",
    r"\b(i can|i could)\b",
    r"\b(i volunteer|i'll take|i can take|i got it)\b",
    r"\b(i'll handle|i can handle|i'll do)\b",
    r"\b(assign me|assign it to me)\b",
    r"\b(done|i did it|i finished)\b",
    r"\b(i promise|i commit|i'm on it|im on it)\b",
]

RE_COMMITMENT = re.compile("|".join(COMMITMENT_PATTERNS), flags=re.IGNORECASE)

class IngestConfig:
    enable_viz: bool = False
    min_text_len: int = 1
    drop_bots: bool = False
    bot_name_hints: Tuple[str, ...] = ("bot", "automod", "mod", "assistant")
    keep_raw_text: bool = True
    strip_code: bool = False          # useful if code is too noisy for your task model
    strip_inline_code: bool = False
    lowercase: bool = True

class BaseIngestor(ABC):
    source: str

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()

    @abstractmethod
    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Return an ingested DataFrame for this source."""
        raise NotImplementedError
        pass

    # ---------------------------
    # Normalization / Cleaning
    # ---------------------------

    def _normalize_schema(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Ensure core keys exist and context is a dict with expected keys.
        """
        df = pd.DataFrame(raw_data)

        # Ensure required columns exist
        required_cols = ["source", "conversation_id", "message_id", "author", "timestamp", "text", "context"]
        for c in required_cols:
            if c not in df.columns:
                df[c] = None

        # Normalize context dict
        def norm_context(c: Any) -> Dict[str, Any]:
            if not isinstance(c, dict):
                c = {}
            return {
                "reply_to": c.get("reply_to"),
                "thread_id": c.get("thread_id"),
                "title": c.get("title"),
            }

        df["context"] = df["context"].apply(norm_context)

        # Enforce source column to match ingestor
        df["source"] = self.source

        # Ensure text is string
        df["text"] = df["text"].fillna("").astype(str)
        df["author"] = df["author"].fillna("unknown").astype(str)

        # Keep original for debugging
        if self.config.keep_raw_text:
            df["raw_text"] = df["text"]

        return df

    def clean_text(self, text: str) -> str:
        t = text

        # remove non-printable control chars
        t = RE_NONPRINT.sub(" ", t)

        # remove URLs
        t = RE_URL.sub("", t)

        # optionally remove code blocks
        if self.config.strip_code:
            t = RE_CODEBLOCK.sub(" ", t)

        if self.config.strip_inline_code:
            t = RE_INLINE_CODE.sub(" ", t)

        # normalize whitespace
        t = RE_MULTI_WS.sub(" ", t).strip()

        # lowercase
        if self.config.lowercase:
            t = t.lower()

        return t

    def _drop_empty_and_short(self, df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = df["text"].astype(str)
        df["text"] = df["text"].apply(self.clean_text)
        df = df[df["text"].str.len() >= self.config.min_text_len]
        return df.reset_index(drop=True)

    def _drop_bots_if_enabled(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_bots:
            return df

        hints = tuple(h.lower() for h in self.config.bot_name_hints)

        def looks_like_bot(author: str) -> bool:
            a = (author or "").lower()
            return any(h in a for h in hints)

        df = df[~df["author"].apply(looks_like_bot)].reset_index(drop=True)
        return df

    # ---------------------------
    # Feature Engineering
    # ---------------------------

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["char_len"] = df["text"].str.len()
        df["word_len"] = df["text"].str.split().str.len()

        df["has_question"] = df["text"].str.contains(r"\?")
        df["has_link"] = df["raw_text"].str.contains(RE_URL) if "raw_text" in df.columns else False

        df["has_commitment_phrase"] = df["text"].str.contains(RE_COMMITMENT)
        df["mentions_user"] = df["raw_text"].str.contains(r"@\w+") if "raw_text" in df.columns else df["text"].str.contains(r"@\w+")

        # simple "action-y" verbs
        df["has_action_verb"] = df["text"].str.contains(r"\b(do|build|fix|ship|deploy|write|review|test|design|make|create|schedule|meet)\b")

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Not all sources have ISO timestamps. Convert carefully.
        # We'll store parsed timestamp as datetime where possible.
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["ts_parsed"] = parsed
        df["ts_date"] = parsed.dt.date.astype("object")
        df["ts_hour"] = parsed.dt.hour.astype("float")
        return df

    # ---------------------------
    # Optional Visualization
    # ---------------------------

    def visualize(self, df: pd.DataFrame, title_suffix: str = "") -> None:
        if not self.config.enable_viz:
            return
        if plt is None:
            print("[WARN] matplotlib not available; skipping visualization.")
            return

        # 1) message length distribution
        plt.figure()
        df["word_len"].hist(bins=30)
        plt.title(f"{self.source}: Word Length Distribution {title_suffix}".strip())
        plt.xlabel("words")
        plt.ylabel("count")
        plt.show()

        # 2) commitment fraction
        plt.figure()
        df["has_commitment_phrase"].value_counts().plot(kind="bar")
        plt.title(f"{self.source}: Commitment Phrase Counts {title_suffix}".strip())
        plt.xlabel("has_commitment_phrase")
        plt.ylabel("count")
        plt.show()

class DiscordIngestor(BaseIngestor):
    source = "discord"

    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = self._normalize_schema(raw_data)
        df = self._drop_bots_if_enabled(df)
        df = self._drop_empty_and_short(df)

        # Discord-specific features
        # - reply chains: context.reply_to present
        df["is_reply"] = df["context"].apply(lambda c: bool(c.get("reply_to")))
        df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id"))
        df["title"] = df["context"].apply(lambda c: c.get("title"))
        df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to"))

        df = self.add_basic_features(df)
        df = self.add_time_features(df)

        self.visualize(df)
        return df

class GitHubIngestor(BaseIngestor):
    source = "github"

    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = self._normalize_schema(raw_data)
        df = self._drop_bots_if_enabled(df)
        df = self._drop_empty_and_short(df)

        # GitHub-specific: issues vs comments
        # In your collector, reply_to = issue id for comments, None for issue itself.
        df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to"))
        df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id"))
        df["title"] = df["context"].apply(lambda c: c.get("title"))
        df["is_issue"] = df["reply_to"].isna()
        df["is_comment"] = ~df["is_issue"]

        # extra features commonly useful
        df["looks_like_bug"] = df["text"].str.contains(r"\bbug|error|exception|crash\b")
        df["looks_like_feature"] = df["text"].str.contains(r"\bfeature|enhancement|request\b")
        df["has_codeblock"] = df["raw_text"].str.contains(RE_CODEBLOCK) if "raw_text" in df.columns else False

        df = self.add_basic_features(df)
        df = self.add_time_features(df)

        self.visualize(df)
        return df


class RedditIngestor(BaseIngestor):
    source = "reddit"

    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = self._normalize_schema(raw_data)
        df = self._drop_bots_if_enabled(df)
        df = self._drop_empty_and_short(df)

        # Reddit-specific
        df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to"))
        df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id"))
        df["title"] = df["context"].apply(lambda c: c.get("title"))
        df["is_comment"] = df["reply_to"].notna()
        df["is_post"] = ~df["is_comment"]

        # detect "advice" or "how to"
        df["is_howto"] = df["text"].str.contains(r"\bhow do i|how to|what should i\b")
        df["is_discussion"] = df["text"].str.contains(r"\bthoughts|opinion|anyone else\b")

        df = self.add_basic_features(df)
        df = self.add_time_features(df)

        self.visualize(df)
        return df


class YouTubeIngestor(BaseIngestor):
    source = "youtube"

    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = self._normalize_schema(raw_data)
        df = self._drop_empty_and_short(df)

        # YouTube transcript segments:
        # Your collector uses timestamp=segment["start"] (float seconds)
        # We'll store it as seconds and also create a minute bucket.
        # Keep the original numeric start in ts_seconds, and also create an ISO-ish column if needed.
        def to_float(x: Any) -> Optional[float]:
            try:
                return float(x)
            except Exception:
                return None

        df["ts_seconds"] = df["timestamp"].apply(to_float)
        df["ts_minute_bucket"] = df["ts_seconds"].apply(lambda s: int(s // 60) if isinstance(s, (int, float)) and pd.notna(s) else None)

        df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to"))
        df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id"))
        df["title"] = df["context"].apply(lambda c: c.get("title"))

        # Transcript-specific: merge very tiny segments optionally (simple heuristic)
        # If you want merging, do it in the combiner so you can tune globally.

        df = self.add_basic_features(df)
        # time features from ISO timestamps don't apply; keep ts_parsed NaT
        df["ts_parsed"] = pd.NaT
        df["ts_date"] = None
        df["ts_hour"] = None

        self.visualize(df)
        return df


class QuoraIngestor(BaseIngestor):
    source = "quora"

    def ingest(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = self._normalize_schema(raw_data)
        df = self._drop_empty_and_short(df)

        # Quora from dataset: text like "Q: ... A: ..."
        df["has_question_prefix"] = df["raw_text"].str.startswith("Q:") if "raw_text" in df.columns else df["text"].str.startswith("q:")
        df["has_answer_prefix"] = df["raw_text"].str.contains(r"\nA:") if "raw_text" in df.columns else df["text"].str.contains(r"\na:")

        df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to"))
        df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id"))
        df["title"] = df["context"].apply(lambda c: c.get("title"))

        df = self.add_basic_features(df)
        df = self.add_time_features(df)

        self.visualize(df)
        return df

class DatasetCombiner:
    """
    Combine ingested datasets into one canonical dataset, de-dup, and add global features.
    """

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()

    def combine(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Flatten context for convenience if present
        if "context" in df.columns:
            # keep context but also expose fields
            if "reply_to" not in df.columns:
                df["reply_to"] = df["context"].apply(lambda c: c.get("reply_to") if isinstance(c, dict) else None)
            if "thread_id" not in df.columns:
                df["thread_id"] = df["context"].apply(lambda c: c.get("thread_id") if isinstance(c, dict) else None)
            if "title" not in df.columns:
                df["title"] = df["context"].apply(lambda c: c.get("title") if isinstance(c, dict) else None)

        # De-dup by (source, message_id)
        if "message_id" in df.columns:
            df = df.drop_duplicates(subset=["source", "message_id"], keep="first")

        # Global features
        df["is_candidate_task"] = (
            df.get("has_action_verb", False).astype(bool)
            | df.get("has_commitment_phrase", False).astype(bool)
        )

        # A slightly stronger "task-like" heuristic:
        df["has_due_signal"] = df["text"].str.contains(r"\b(today|tomorrow|by\b|deadline|before\b|next week|this week)\b", regex=True)
        df["mentions_artifact"] = df["text"].str.contains(r"\b(doc|document|pr|pull request|issue|ticket|meeting|call|zoom|calendar)\b", regex=True)

        df["is_strong_task_candidate"] = df["is_candidate_task"] & (df["has_due_signal"] | df["mentions_artifact"])

        df = df.reset_index(drop=True)

        # Optional viz on combined
        if self.config.enable_viz and plt is not None and not df.empty:
            plt.figure()
            df["source"].value_counts().plot(kind="bar")
            plt.title("Combined: Samples per Source")
            plt.xlabel("source")
            plt.ylabel("count")
            plt.show()

        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        # choose format by extension
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        elif path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            # default
            df.to_parquet(path + ".parquet", index=False)



def ingest_all(
    discord_raw: Optional[List[Dict[str, Any]]] = None,
    github_raw: Optional[List[Dict[str, Any]]] = None,
    reddit_raw: Optional[List[Dict[str, Any]]] = None,
    youtube_raw: Optional[List[Dict[str, Any]]] = None,
    quora_raw: Optional[List[Dict[str, Any]]] = None,
    config: Optional[IngestConfig] = None,
) -> pd.DataFrame:
    """
    Ingest multiple sources and return one combined DataFrame.
    """
    cfg = config or IngestConfig()
    dfs: List[pd.DataFrame] = []

    if discord_raw:
        dfs.append(DiscordIngestor(cfg).ingest(discord_raw))
    if github_raw:
        dfs.append(GitHubIngestor(cfg).ingest(github_raw))
    if reddit_raw:
        dfs.append(RedditIngestor(cfg).ingest(reddit_raw))
    if youtube_raw:
        dfs.append(YouTubeIngestor(cfg).ingest(youtube_raw))
    if quora_raw:
        dfs.append(QuoraIngestor(cfg).ingest(quora_raw))

    return DatasetCombiner(cfg).combine(dfs)
        
