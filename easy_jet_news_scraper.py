import requests
from bs4 import BeautifulSoup
import pandas as pd
import requests
import xml.etree.ElementTree as ET


class EasyJetNewsScraper:
    def __init__(self):
        # pure RSS‐feed URLs
        self.urls = {
            "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EZJ.L&region=GB&lang=en-GB",
            "Google News":   "https://news.google.com/rss/search?q=EZJ.L&hl=en-GB&gl=GB&ceid=GB:en",
            "Finviz":        "https://finviz.com/rss.ashx?t=EZJ",
        }
        self.fetch_methods = {
            "Yahoo Finance": self.fetch_yahoo,
            "Google News":   self.fetch_google,
            "Finviz":        self.fetch_finviz,
        }

    def fetch_yahoo(self) -> pd.DataFrame:
        """Fetch via Yahoo Finance RSS."""
        return self._parse_rss(self.urls["Yahoo Finance"])

    def fetch_google(self) -> pd.DataFrame:
        """Fetch via Google News RSS."""
        return self._parse_rss(self.urls["Google News"])

    def fetch_finviz(self) -> pd.DataFrame:
        """Fetch via Finviz RSS."""
        return self._parse_rss(self.urls["Finviz"])

    def _parse_rss(self, url: str) -> pd.DataFrame:
        """Generic RSS parser—pulls title, link, pubDate."""
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.find("channel").findall("item"):
            items.append({
                "Date":     item.findtext("pubDate", default=""),
                "Headline": item.findtext("title",   default=""),
                "Link":     item.findtext("link",    default=""),
            })
        return pd.DataFrame(items)
