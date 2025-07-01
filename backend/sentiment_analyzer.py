import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import re
import logging
from collections import defaultdict
import yfinance as yf

logger = logging.getLogger(__name__)

class SocialSentimentAnalyzer:
    """
    Analyzes social media sentiment from Reddit and other sources
    """
    
    def __init__(self, reddit_client_id=None, reddit_client_secret=None):
        # Initialize Reddit API (you'll need to get these credentials)
        if reddit_client_id and reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent='StockSentimentAnalyzer/1.0'
            )
        else:
            self.reddit = None
            logger.warning("Reddit API not configured - using mock data")
        
        # Popular investment subreddits
        self.subreddits = [
            'wallstreetbets', 'stocks', 'investing', 'StockMarket',
            'options', 'SecurityAnalysis', 'ValueInvesting'
        ]
        
        # Sentiment keywords and weights
        self.bullish_keywords = {
            'moon': 3, 'rocket': 3, 'bull': 2, 'calls': 2, 'buy': 2,
            'long': 2, 'up': 1, 'green': 1, 'gains': 2, 'yolo': 3,
            'diamond hands': 3, 'hold': 2, 'buying': 2, 'bullish': 3,
            'squeeze': 3, 'pump': 2, 'breakout': 2, 'support': 1
        }
        
        self.bearish_keywords = {
            'put': 2, 'puts': 2, 'short': 2, 'sell': 2, 'down': 1,
            'bear': 2, 'red': 1, 'crash': 3, 'dump': 3, 'overvalued': 2,
            'bubble': 3, 'bearish': 3, 'resistance': 1, 'overbought': 2,
            'dead': 2, 'rip': 2, 'bag': 2, 'bagholder': 3
        }
    
    def analyze_symbol_sentiment(self, symbol, lookback_hours=24):
        """
        Analyze sentiment for a specific symbol across social media
        """
        try:
            if self.reddit:
                reddit_sentiment = self.analyze_reddit_sentiment(symbol, lookback_hours)
            else:
                reddit_sentiment = self.generate_mock_sentiment(symbol)
            
            # Aggregate sentiment data
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'lookback_hours': lookback_hours,
                'reddit_metrics': reddit_sentiment,
                'overall_sentiment': self.calculate_overall_sentiment(reddit_sentiment),
                'sentiment_signals': self.generate_sentiment_signals(reddit_sentiment),
                'risk_indicators': self.identify_risk_indicators(reddit_sentiment)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return None
    
    def analyze_reddit_sentiment(self, symbol, lookback_hours):
        """
        Analyze Reddit sentiment for a symbol
        """
        sentiment_data = {
            'total_mentions': 0,
            'posts_analyzed': 0,
            'comments_analyzed': 0,
            'average_sentiment': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'mention_velocity': 0,
            'top_posts': [],
            'keyword_scores': {'bullish': 0, 'bearish': 0},
            'wsb_specific': {'rocket_count': 0, 'yolo_count': 0, 'gain_porn': 0, 'loss_porn': 0}
        }
        
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts mentioning the symbol
                for submission in subreddit.search(symbol, time_filter='day', limit=50):
                    if datetime.utcfromtimestamp(submission.created_utc) < cutoff_time:
                        continue
                    
                    # Analyze post
                    post_sentiment = self.analyze_text_sentiment(submission.title + ' ' + submission.selftext)
                    sentiment_data['posts_analyzed'] += 1
                    
                    # Track top posts
                    if submission.score > 100:
                        sentiment_data['top_posts'].append({
                            'title': submission.title,
                            'score': submission.score,
                            'sentiment': post_sentiment['sentiment'],
                            'url': f"reddit.com{submission.permalink}"
                        })
                    
                    # Analyze comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:20]:  # Limit comments per post
                        comment_sentiment = self.analyze_text_sentiment(comment.body)
                        sentiment_data['comments_analyzed'] += 1
                        
                        # Update sentiment distribution
                        if comment_sentiment['sentiment'] > 0.1:
                            sentiment_data['sentiment_distribution']['positive'] += 1
                        elif comment_sentiment['sentiment'] < -0.1:
                            sentiment_data['sentiment_distribution']['negative'] += 1
                        else:
                            sentiment_data['sentiment_distribution']['neutral'] += 1
                        
                        # Update keyword scores
                        sentiment_data['keyword_scores']['bullish'] += comment_sentiment['bullish_score']
                        sentiment_data['keyword_scores']['bearish'] += comment_sentiment['bearish_score']
                        
                        # WSB specific tracking
                        text_lower = comment.body.lower()
                        sentiment_data['wsb_specific']['rocket_count'] += text_lower.count('🚀')
                        sentiment_data['wsb_specific']['yolo_count'] += text_lower.count('yolo')
                        if 'gain' in text_lower and 'porn' in text_lower:
                            sentiment_data['wsb_specific']['gain_porn'] += 1
                        if 'loss' in text_lower and 'porn' in text_lower:
                            sentiment_data['wsb_specific']['loss_porn'] += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {subreddit_name}: {e}")
                continue
        
        # Calculate aggregates
        total_items = sentiment_data['posts_analyzed'] + sentiment_data['comments_analyzed']
        sentiment_data['total_mentions'] = total_items
        
        if total_items > 0:
            total_sentiment = sum(sentiment_data['sentiment_distribution'].values())
            if total_sentiment > 0:
                sentiment_data['average_sentiment'] = (
                    (sentiment_data['sentiment_distribution']['positive'] - 
                     sentiment_data['sentiment_distribution']['negative']) / total_sentiment
                )
        
        # Calculate mention velocity (mentions per hour)
        sentiment_data['mention_velocity'] = total_items / lookback_hours
        
        # Sort top posts by score
        sentiment_data['top_posts'] = sorted(
            sentiment_data['top_posts'], 
            key=lambda x: x['score'], 
            reverse=True
        )[:5]
        
        return sentiment_data
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob and keyword analysis
        """
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text_lower = text.lower()
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Keyword-based sentiment
        bullish_score = sum(
            text_lower.count(keyword) * weight 
            for keyword, weight in self.bullish_keywords.items()
        )
        
        bearish_score = sum(
            text_lower.count(keyword) * weight 
            for keyword, weight in self.bearish_keywords.items()
        )
        
        # Combined sentiment
        keyword_sentiment = (bullish_score - bearish_score) / max(bullish_score + bearish_score, 1)
        combined_sentiment = (polarity + keyword_sentiment) / 2
        
        return {
            'sentiment': combined_sentiment,
            'polarity': polarity,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }
    
    def generate_mock_sentiment(self, symbol):
        """
        Generate mock sentiment data for testing without Reddit API
        """
        # Generate realistic-looking mock data
        np.random.seed(hash(symbol) % 1000)  # Consistent per symbol
        
        base_mentions = np.random.randint(50, 500)
        sentiment_skew = np.random.normal(0, 0.3)  # Slight random bias
        
        positive = int(base_mentions * (0.4 + sentiment_skew))
        negative = int(base_mentions * (0.4 - sentiment_skew))
        neutral = base_mentions - positive - negative
        
        return {
            'total_mentions': base_mentions,
            'posts_analyzed': int(base_mentions * 0.2),
            'comments_analyzed': int(base_mentions * 0.8),
            'average_sentiment': sentiment_skew,
            'sentiment_distribution': {
                'positive': max(0, positive),
                'neutral': max(0, neutral),
                'negative': max(0, negative)
            },
            'mention_velocity': base_mentions / 24,
            'top_posts': [
                {
                    'title': f"${symbol} to the moon! 🚀🚀🚀",
                    'score': np.random.randint(100, 1000),
                    'sentiment': 0.8,
                    'url': 'reddit.com/r/wallstreetbets/mock'
                }
            ],
            'keyword_scores': {
                'bullish': np.random.randint(10, 100),
                'bearish': np.random.randint(5, 50)
            },
            'wsb_specific': {
                'rocket_count': np.random.randint(0, 50),
                'yolo_count': np.random.randint(0, 20),
                'gain_porn': np.random.randint(0, 10),
                'loss_porn': np.random.randint(0, 5)
            }
        }
    
    def calculate_overall_sentiment(self, reddit_data):
        """
        Calculate overall sentiment score and interpretation
        """
        avg_sentiment = reddit_data['average_sentiment']
        mention_velocity = reddit_data['mention_velocity']
        
        # Normalize sentiment to -100 to 100 scale
        sentiment_score = avg_sentiment * 100
        
        # Adjust for mention velocity (high activity = stronger signal)
        if mention_velocity > 20:  # High activity
            sentiment_score *= 1.2
        elif mention_velocity < 5:  # Low activity
            sentiment_score *= 0.8
        
        # Cap at reasonable bounds
        sentiment_score = max(-100, min(100, sentiment_score))
        
        # Interpretation
        if sentiment_score > 50:
            interpretation = "Very Bullish"
        elif sentiment_score > 20:
            interpretation = "Bullish"
        elif sentiment_score > -20:
            interpretation = "Neutral"
        elif sentiment_score > -50:
            interpretation = "Bearish"
        else:
            interpretation = "Very Bearish"
        
        return {
            'score': sentiment_score,
            'interpretation': interpretation,
            'confidence': self.calculate_confidence(reddit_data)
        }
    
    def calculate_confidence(self, reddit_data):
        """
        Calculate confidence in sentiment reading
        """
        total_mentions = reddit_data['total_mentions']
        
        if total_mentions > 1000:
            return "Very High"
        elif total_mentions > 500:
            return "High"
        elif total_mentions > 100:
            return "Moderate"
        elif total_mentions > 50:
            return "Low"
        else:
            return "Very Low"
    
    def generate_sentiment_signals(self, reddit_data):
        """
        Generate actionable signals from sentiment data
        """
        signals = []
        
        # Check for extreme sentiment
        avg_sentiment = reddit_data['average_sentiment']
        if avg_sentiment > 0.5:
            signals.append({
                'type': 'extreme_bullish',
                'message': 'Extreme bullish sentiment detected - possible FOMO',
                'severity': 'warning'
            })
        elif avg_sentiment < -0.5:
            signals.append({
                'type': 'extreme_bearish',
                'message': 'Extreme bearish sentiment - potential capitulation',
                'severity': 'warning'
            })
        
        # Check mention velocity
        velocity = reddit_data['mention_velocity']
        if velocity > 50:
            signals.append({
                'type': 'high_activity',
                'message': f'Unusually high social activity ({velocity:.0f} mentions/hour)',
                'severity': 'alert'
            })
        
        # Check WSB specific signals
        wsb = reddit_data['wsb_specific']
        if wsb['rocket_count'] > 20:
            signals.append({
                'type': 'meme_stock_alert',
                'message': 'High meme stock activity detected',
                'severity': 'warning'
            })
        
        if wsb['yolo_count'] > 10:
            signals.append({
                'type': 'yolo_trades',
                'message': 'High-risk YOLO trading detected',
                'severity': 'alert'
            })
        
        return signals
    
    def identify_risk_indicators(self, reddit_data):
        """
        Identify specific risk indicators from sentiment
        """
        risks = []
        
        # Sentiment distribution analysis
        dist = reddit_data['sentiment_distribution']
        total = sum(dist.values())
        
        if total > 0:
            positive_ratio = dist['positive'] / total
            negative_ratio = dist['negative'] / total
            
            # Check for herd mentality
            if positive_ratio > 0.8:
                risks.append({
                    'type': 'herd_bullish',
                    'description': 'Overwhelming bullish consensus - contrarian indicator',
                    'risk_level': 'high'
                })
            elif negative_ratio > 0.8:
                risks.append({
                    'type': 'herd_bearish',
                    'description': 'Overwhelming bearish consensus - potential bottom',
                    'risk_level': 'medium'
                })
        
        # Check for pump indicators
        if reddit_data['keyword_scores']['bullish'] > 200:
            risks.append({
                'type': 'pump_risk',
                'description': 'Excessive bullish keywords - possible pump',
                'risk_level': 'high'
            })
        
        # YOLO risk
        if reddit_data['wsb_specific']['yolo_count'] > 20:
            risks.append({
                'type': 'retail_speculation',
                'description': 'High retail speculation detected',
                'risk_level': 'medium'
            })
        
        return risks
    
    def get_trending_tickers(self, subreddit='wallstreetbets', limit=10):
        """
        Get currently trending tickers from Reddit
        """
        if not self.reddit:
            # Return mock trending tickers
            return ['GME', 'AMC', 'TSLA', 'AAPL', 'SPY', 'NVDA', 'PLTR', 'BB', 'NOK', 'AMD']
        
        ticker_counts = defaultdict(int)
        
        try:
            subreddit = self.reddit.subreddit(subreddit)
            
            # Scan hot posts
            for submission in subreddit.hot(limit=100):
                # Extract potential tickers (1-5 uppercase letters)
                tickers = re.findall(r'\b[A-Z]{1,5}\b', submission.title + ' ' + submission.selftext)
                
                for ticker in tickers:
                    # Filter out common words
                    if ticker not in ['I', 'A', 'THE', 'TO', 'OF', 'AND', 'OR', 'FOR', 'IN', 'ON', 'AT']:
                        # Verify it's a real ticker using yfinance
                        try:
                            stock = yf.Ticker(ticker)
                            if stock.info.get('regularMarketPrice'):
                                ticker_counts[ticker] += submission.score  # Weight by post score
                        except:
                            pass
            
            # Sort by count and return top N
            sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
            return [ticker for ticker, count in sorted_tickers[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return []