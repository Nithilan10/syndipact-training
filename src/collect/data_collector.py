from abc import ABC, abstractmethod
import pandas as pd
import requests
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi

class DataCollector(ABC):
    @abstractmethod
    def collect(self):
        """
        Returns a list of normalized message objects
        """
        pass


class DiscordCollector(DataCollector):
    BASE_URL = "https://discord.com/api/v10"

    def __init__(self, bot_token: str, channel_id: str):
        self.channel_id = channel_id
        self.headers = {
            "Authorization": f"Bot {bot_token}"
        }

    def collect(self, limit=100):
        url = f"{self.BASE_URL}/channels/{self.channel_id}/messages"
        res = requests.get(url, headers=self.headers, params={"limit": limit})
        res.raise_for_status()

        messages = res.json()
        data = []

        for msg in messages:
            data.append({
                "source": "discord",
                "conversation_id": self.channel_id,
                "message_id": msg["id"],
                "author": msg["author"]["username"],
                "timestamp": msg["timestamp"],
                "text": msg["content"],
                "context": {
                    "reply_to": msg.get("referenced_message", {}).get("id"),
                    "thread_id": None,
                    "title": None
                }
            })

        return data


class GitHubCollector(DataCollector):
    BASE_URL = "https://api.github.com"

    def __init__(self, api_key: str, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/vnd.github+json"
        }

    def _get(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        res = requests.get(url, headers=self.headers, params=params)
        res.raise_for_status()
        return res.json()

    def collect(self):
        data = []

        # Issues
        issues = self._get(
            f"/repos/{self.owner}/{self.repo}/issues",
            params={"state": "all", "per_page": 100}
        )

        for issue in issues:
            if "pull_request" in issue:
                continue

            data.append({
                "source": "github",
                "conversation_id": issue["number"],
                "message_id": issue["id"],
                "author": issue["user"]["login"],
                "timestamp": issue["created_at"],
                "text": issue["title"] + "\n" + (issue["body"] or ""),
                "context": {
                    "reply_to": None,
                    "thread_id": issue["number"],
                    "title": issue["title"]
                }
            })

            # Issue comments
            comments = self._get(issue["comments_url"])
            for c in comments:
                data.append({
                    "source": "github",
                    "conversation_id": issue["number"],
                    "message_id": c["id"],
                    "author": c["user"]["login"],
                    "timestamp": c["created_at"],
                    "text": c["body"],
                    "context": {
                        "reply_to": issue["id"],
                        "thread_id": issue["number"],
                        "title": issue["title"]
                    }
                })

        return data

    class RedditCollector(DataCollector):
        BASE_URL = "https://oauth.reddit.com"

        def __init__(self, access_token: str, subreddit: str, limit=50):
            self.subreddit = subreddit
            self.limit = limit
            self.headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "SyndipactAI/0.1"
            }

        def collect(self):
            url = f"{self.BASE_URL}/r/{self.subreddit}/new"
            res = requests.get(url, headers=self.headers, params={"limit": self.limit})
            res.raise_for_status()

            posts = res.json()["data"]["children"]
            data = []

            for post in posts:
                p = post["data"]

                # Main post
                data.append({
                    "source": "reddit",
                    "conversation_id": p["id"],
                    "message_id": p["name"],
                    "author": p["author"],
                    "timestamp": datetime.utcfromtimestamp(p["created_utc"]).isoformat(),
                    "text": f"{p['title']}\n{p.get('selftext', '')}",
                    "context": {
                        "reply_to": None,
                        "thread_id": p["id"],
                        "title": p["title"]
                    }
                })

                # Fetch comments
                comments_url = f"{self.BASE_URL}/comments/{p['id']}"
                comments_res = requests.get(comments_url, headers=self.headers)
                comments_res.raise_for_status()

                comments = comments_res.json()[1]["data"]["children"]

                for c in comments:
                    if c["kind"] != "t1":
                        continue

                    cd = c["data"]
                    data.append({
                        "source": "reddit",
                        "conversation_id": p["id"],
                        "message_id": cd["name"],
                        "author": cd["author"],
                        "timestamp": datetime.utcfromtimestamp(cd["created_utc"]).isoformat(),
                        "text": cd["body"],
                        "context": {
                            "reply_to": cd.get("parent_id"),
                            "thread_id": p["id"],
                            "title": p["title"]
                        }
                    })

            return data
        
    class YouTubeCollector(DataCollector):
            def __init__(self, video_id: str):
                self.video_id = video_id

            def collect(self):
                transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
                data = []

                for i, segment in enumerate(transcript):
                    data.append({
                        "source": "youtube",
                        "conversation_id": self.video_id,
                        "message_id": f"{self.video_id}_{i}",
                        "author": "speaker",
                        "timestamp": segment["start"],
                        "text": segment["text"],
                        "context": {
                            "reply_to": None,
                            "thread_id": None,
                            "title": None
                        }
                    })

                return data
    class QuoraCollector(DataCollector):
        def __init__(self, dataset_path: str):
            self.dataset_path = dataset_path

        def collect(self):
            df = pd.read_csv(self.dataset_path)
            data = []

            for i, row in df.iterrows():
                data.append({
                    "source": "quora",
                    "conversation_id": f"quora_{i}",
                    "message_id": f"q_{i}",
                    "author": "unknown",
                    "timestamp": None,
                    "text": f"Q: {row['question']} \nA: {row['answer']}",
                    "context": {
                        "reply_to": None,
                        "thread_id": f"quora_{i}",
                        "title": row["question"]
                    }
                })

            return data

        