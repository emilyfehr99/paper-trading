import os
import logging
from typing import Any, Dict
import finnhub

log = logging.getLogger("SentimentEngine")

class SentimentEngine:
    """
    Analyzes market sentiment using Finnhub and Reddit (PRAW).
    Provides 'Hype' and 'Sentiment' features to help the model avoid traps.
    """
    def __init__(self, finnhub_api_key: str | None = None):
        self.api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.client = finnhub.Client(api_key=self.api_key) if self.api_key else None
        
        # Reddit PRAW Credentials
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "AetherisBot/1.0")

    def get_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Fetches combined news and social sentiment.
        """
        sentiment = {"sentiment_score": 0.5, "buzz_score": 0.0, "reddit_hype": 0.0}
        
        # 1. Finnhub News Sentiment
        if self.client:
            try:
                res = self.client.news_sentiment(symbol)
                if res:
                    sentiment["sentiment_score"] = float(res.get("sentiment", {}).get("bullishPercent", 50.0)) / 100.0
                    sentiment["buzz_score"] = float(res.get("buzz", {}).get("articlesInLastWeek", 0.0))
            except Exception as e:
                log.debug(f"Finnhub sentiment failed for {symbol}: {e}")

        # 2. Reddit Hype (Placeholder for PRAW implementation)
        sentiment["reddit_hype"] = self.get_reddit_hype(symbol)
        
        return sentiment

    def get_reddit_hype(self, symbol: str) -> float:
        """
        Heuristic for Reddit hype. High value indicates retail clustering.
        """
        if not self.reddit_client_id:
            return 0.0 # No keys, no hype detection
            
        try:
            # Note: We'd use a background worker for actual scraping 
            # to avoid blocking the main trading loop.
            return 0.05 # Mock for now
        except Exception:
            return 0.0

    def get_alpaca_news_sentiment(self, symbol: str, texts: list[str]) -> float:
        """
        Calculates a lexicon-based sentiment score from -1.0 to 1.0 based on combined headline and summary content.
        """
        if not texts:
            return 0.0
            
        pos_words = {"beat", "upgrade", "surge", "growth", "profit", "win", "bullish", "success", "expand", "raise", "buy"}
        neg_words = {"miss", "downgrade", "crash", "loss", "fail", "bankruptcy", "probe", "investigation", "warning", "drop", "sell", "lawsuit", "halt"}
        
        score = 0.0
        for text in texts:
            text_lower = text.lower()
            pos_matches = sum(1 for w in pos_words if w in text_lower)
            neg_matches = sum(1 for w in neg_words if w in text_lower)
            
            diff = pos_matches - neg_matches
            if diff > 0:
                score += 0.25 * diff
            elif diff < 0:
                score += 0.25 * diff
                
        return max(-1.0, min(1.0, score))
