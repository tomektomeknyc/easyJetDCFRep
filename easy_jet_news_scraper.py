import requests
import pandas as pd
import xml.etree.ElementTree as ET
import re

class EasyJetNewsScraper:
    """
    Scrapes multiple RSS feeds and returns only items
    that mention EZJ.L or easyJet in the headline.
    """
    def __init__(self):
        self.urls = {
            "Yahoo Finance":    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EZJ.L&region=GB&lang=en-GB",
            "Google News":      "https://news.google.com/rss/search?q=EZJ.L&hl=en-GB&gl=GB&ceid=GB:en",
            "Finviz":           "https://finviz.com/rss.ashx?t=EZJ",
            "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
            "FT Headlines":     "https://www.ft.com/?format=rss"
        }
        self.fetch_methods = {
            name: getattr(self, f"fetch_{self._slug(name)}")
            for name in self.urls
        }

    def _slug(self, name: str) -> str:
        # turn "Yahoo Finance" into "yahoo_finance", etc.
        return re.sub(r"[^\w]+", "_", name.strip().lower())

    def fetch_yahoo_finance(self) -> pd.DataFrame:
        return self._parse_and_filter("Yahoo Finance")

    def fetch_google_news(self) -> pd.DataFrame:
        return self._parse_and_filter("Google News")

    def fetch_finviz(self) -> pd.DataFrame:
        return self._parse_and_filter("Finviz")

    def fetch_reuters_business(self) -> pd.DataFrame:
        return self._parse_and_filter("Reuters Business")

    def fetch_ft_headlines(self) -> pd.DataFrame:
        return self._parse_and_filter("FT Headlines")

    def _parse_and_filter(self, source: str) -> pd.DataFrame:
        """
        1) fetch RSS from self.urls[source]
        2) parse <item> elements
        3) filter to keep only those whose title contains EZJ.L or easyJet
        """
        url = self.urls[source]
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        items = []
        for itm in root.find("channel").findall("item"):
            title = itm.findtext("title", default="")
            link  = itm.findtext("link",  default="")
            date  = itm.findtext("pubDate",default="")
            # only keep containing EZJ.L or easyJet
            if re.search(r"\bEZJ\.L\b", title, re.IGNORECASE) or "easyJet" in title:
                items.append({
                    "Date":     date,
                    "Headline": title,
                    "Link":     link,
                    "Source":   source
                })

        return pd.DataFrame(items)
