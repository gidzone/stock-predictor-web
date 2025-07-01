import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from volatility_predictor import VolatilityPredictor
from sentiment_analyzer import SocialSentimentAnalyzer

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    EXTREME = 5

@dataclass
class PortfolioAlert:
    symbol: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    data: Dict
    action_required: bool = False

class PortfolioGuardian:
    """
    Main class that combines all risk analysis components
    Monitors portfolio for risks and generates actionable alerts
    """
    
    def __init__(self, api_key, secret_key, base_url, reddit_credentials=None):
        # Initialize components
        self.volatility_predictor = VolatilityPredictor(api_key, secret_key, base_url)
        self.sentiment_analyzer = SocialSentimentAnalyzer(
            reddit_client_id=reddit_credentials.get('client_id') if reddit_credentials else None,
            reddit_client_secret=reddit_credentials.get('client_secret') if reddit_credentials else None
        )
        
        # Alert thresholds
        self.thresholds = {
            'volatility_spike': 1.5,  # 50% increase in volatility
            'drawdown_warning': -0.05,  # 5% drawdown prediction
            'sentiment_extreme': 70,  # Sentiment score above 70 or below -70
            'mention_velocity_spike': 3.0,  # 3x normal mention rate
        }
        
        # Portfolio tracking
        self.monitored_positions = {}
        self.alerts_history = []
        
    def analyze_position(self, symbol: str, position_size: float = None, 
                    purchase_price: float = None) -> Dict:
        """
        Comprehensive analysis of a single position
        """
        try:
            # Initialize default response structure
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'risk_analysis': None,
                'sentiment_analysis': None,
                'combined_risk_score': {'score': 50, 'level': 'UNKNOWN'},
                'alerts': [],
                'recommendations': []
            }
        
            # Try to get volatility predictions
            try:
                risk_metrics = self.volatility_predictor.predict_risk_metrics(symbol)
                analysis['risk_analysis'] = risk_metrics
            except Exception as e:
                logger.warning(f"Could not get risk metrics for {symbol}: {e}")
                # Add a default risk analysis structure
                analysis['risk_analysis'] = {
                    'symbol': symbol,
                    'current_price': 0,
                    'current_volatility': 0,
                    'predicted_volatility_5d': 0,
                    'predicted_volatility_20d': 0,
                    'predicted_max_drawdown_5d': 0,
                    'predicted_max_drawdown_20d': 0,
                    'value_at_risk_5d': 0,
                    'value_at_risk_20d': 0,
                    'risk_score': 50,
                    'risk_level': 'UNKNOWN',
                    'volatility_trend': 'unknown',
                    'risk_percentile': 50
                }
            
            # Try to get sentiment analysis
            try:
                sentiment_data = self.sentiment_analyzer.analyze_symbol_sentiment(symbol)
                analysis['sentiment_analysis'] = sentiment_data
            except Exception as e:
                logger.warning(f"Could not get sentiment for {symbol}: {e}")
                # Add a default sentiment structure
                analysis['sentiment_analysis'] = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'lookback_hours': 24,
                    'reddit_metrics': {
                        'total_mentions': 0,
                        'mention_velocity': 0,
                        'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
                    },
                    'overall_sentiment': {
                        'score': 0,
                        'interpretation': 'Unknown',
                        'confidence': 'Very Low'
                    },
                    'sentiment_signals': [],
                    'risk_indicators': []
                }
            
            # Only calculate combined risk score if we have valid data
            if analysis['risk_analysis'] and analysis['sentiment_analysis']:
                analysis['combined_risk_score'] = self._calculate_combined_risk_score(
                    analysis['risk_analysis'], 
                    analysis['sentiment_analysis']
                )
                analysis['alerts'] = self._generate_position_alerts(
                    symbol, 
                    analysis['risk_analysis'], 
                    analysis['sentiment_analysis']
                )
                analysis['recommendations'] = self._generate_recommendations(
                    analysis['risk_analysis'], 
                    analysis['sentiment_analysis']
                )
            
            # Add position-specific metrics if provided
            if position_size and purchase_price and analysis['risk_analysis']:
                current_price = analysis['risk_analysis'].get('current_price', 0)
                if current_price > 0:
                    position_value = position_size * current_price
                    pnl = (current_price - purchase_price) * position_size
                    pnl_pct = (current_price / purchase_price - 1) * 100
                    
                    analysis['position_metrics'] = {
                        'position_size': position_size,
                        'position_value': position_value,
                        'purchase_price': purchase_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_percentage': pnl_pct,
                        'position_var_5d': analysis['risk_analysis'].get('value_at_risk_5d', 0) * position_size,
                        'position_var_20d': analysis['risk_analysis'].get('value_at_risk_20d', 0) * position_size
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing position {symbol}: {e}")
            # Return a minimal valid structure
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'risk_analysis': None,
                'sentiment_analysis': None,
                'combined_risk_score': {'score': 50, 'level': 'UNKNOWN'},
                'alerts': [],
                'recommendations': [],
                'error': str(e)
            }
    
    def monitor_portfolio(self, portfolio: List[Dict]) -> Dict:
        """
        Monitor entire portfolio for risks
        portfolio: List of {'symbol': str, 'shares': float, 'purchase_price': float}
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_analysis': [],
            'portfolio_metrics': {},
            'alerts': [],
            'trending_risks': [],
            'action_items': []
        }
        
        total_value = 0
        total_var_5d = 0
        total_var_20d = 0
        
        for position in portfolio:
            # Analyze each position
            analysis = self.analyze_position(
                position['symbol'],
                position.get('shares'),
                position.get('purchase_price')
            )
            
            if analysis:
                results['portfolio_analysis'].append(analysis)
                
                # Aggregate metrics
                if 'position_metrics' in analysis:
                    total_value += analysis['position_metrics']['position_value']
                    total_var_5d += analysis['position_metrics']['position_var_5d']
                    total_var_20d += analysis['position_metrics']['position_var_20d']
                
                # Collect alerts
                results['alerts'].extend(analysis['alerts'])
        
        # Calculate portfolio-level metrics
        results['portfolio_metrics'] = {
            'total_value': total_value,
            'total_var_5d': total_var_5d,
            'total_var_20d': total_var_20d,
            'var_percentage_5d': (total_var_5d / total_value * 100) if total_value > 0 else 0,
            'var_percentage_20d': (total_var_20d / total_value * 100) if total_value > 0 else 0,
            'diversification_score': self._calculate_diversification_score(results['portfolio_analysis'])
        }
        
        # Identify trending risks across portfolio
        results['trending_risks'] = self._identify_trending_risks(results['portfolio_analysis'])
        
        # Generate action items
        results['action_items'] = self._generate_action_items(results)
        
        return results
    
    def _calculate_combined_risk_score(self, risk_metrics: Dict, sentiment_data: Dict) -> Dict:
        """
        Combine volatility and sentiment into unified risk score
        """
        if not risk_metrics or not sentiment_data:
            return {'score': 50, 'level': 'UNKNOWN'}
        
        # Volatility component (0-100)
        vol_score = risk_metrics.get('risk_score', 50)
        
        # Sentiment component (convert -100 to 100 into risk score)
        sentiment_score = sentiment_data.get('overall_sentiment', {}).get('score', 0)
        
        # Extreme sentiment (both very bullish and bearish) increases risk
        sentiment_risk = abs(sentiment_score) * 0.7  # 0-70 range
        
        # If sentiment is extremely bullish, add extra risk
        if sentiment_score > 70:
            sentiment_risk += 20  # FOMO risk
        
        # Combine scores (weighted average)
        combined_score = (vol_score * 0.6 + sentiment_risk * 0.4)
        
        # Determine risk level
        if combined_score < 20:
            level = RiskLevel.VERY_LOW
        elif combined_score < 40:
            level = RiskLevel.LOW
        elif combined_score < 60:
            level = RiskLevel.MODERATE
        elif combined_score < 80:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.EXTREME
        
        return {
            'score': combined_score,
            'level': level.name,
            'components': {
                'volatility_risk': vol_score,
                'sentiment_risk': sentiment_risk
            }
        }
    
    def _generate_position_alerts(self, symbol: str, risk_metrics: Dict, 
                                 sentiment_data: Dict) -> List[PortfolioAlert]:
        """
        Generate alerts for a position based on various criteria
        """
        alerts = []
        
        # Volatility alerts
        if risk_metrics:
            # Check for volatility spike
            if risk_metrics['predicted_volatility_5d'] > risk_metrics['current_volatility'] * self.thresholds['volatility_spike']:
                alerts.append(PortfolioAlert(
                    symbol=symbol,
                    alert_type='volatility_spike',
                    severity=AlertSeverity.WARNING,
                    message=f"Volatility expected to increase {risk_metrics['predicted_volatility_5d']/risk_metrics['current_volatility']:.1f}x in next 5 days",
                    timestamp=datetime.now(),
                    data={'current_vol': risk_metrics['current_volatility'], 
                          'predicted_vol': risk_metrics['predicted_volatility_5d']},
                    action_required=True
                ))
            
            # Check for drawdown risk
            if risk_metrics['predicted_max_drawdown_5d'] < self.thresholds['drawdown_warning'] * 100:
                alerts.append(PortfolioAlert(
                    symbol=symbol,
                    alert_type='drawdown_risk',
                    severity=AlertSeverity.CRITICAL if risk_metrics['predicted_max_drawdown_5d'] < -10 else AlertSeverity.WARNING,
                    message=f"Potential drawdown of {abs(risk_metrics['predicted_max_drawdown_5d']):.1f}% predicted",
                    timestamp=datetime.now(),
                    data={'predicted_drawdown': risk_metrics['predicted_max_drawdown_5d']},
                    action_required=True
                ))
        
        # Sentiment alerts
        if sentiment_data and sentiment_data['overall_sentiment']:
            sentiment_score = sentiment_data['overall_sentiment']['score']
            
            # Extreme sentiment
            if abs(sentiment_score) > self.thresholds['sentiment_extreme']:
                sentiment_type = "bullish" if sentiment_score > 0 else "bearish"
                alerts.append(PortfolioAlert(
                    symbol=symbol,
                    alert_type=f'extreme_{sentiment_type}_sentiment',
                    severity=AlertSeverity.WARNING,
                    message=f"Extreme {sentiment_type} sentiment detected ({sentiment_score:.0f}/100)",
                    timestamp=datetime.now(),
                    data={'sentiment_score': sentiment_score},
                    action_required=sentiment_score > 80  # Action required for extreme bullish (FOMO)
                ))
            
            # Mention velocity spike
            if sentiment_data['reddit_metrics']['mention_velocity'] > 20:
                alerts.append(PortfolioAlert(
                    symbol=symbol,
                    alert_type='social_activity_spike',
                    severity=AlertSeverity.INFO,
                    message=f"High social media activity: {sentiment_data['reddit_metrics']['mention_velocity']:.0f} mentions/hour",
                    timestamp=datetime.now(),
                    data={'mention_velocity': sentiment_data['reddit_metrics']['mention_velocity']},
                    action_required=False
                ))
            
            # Meme stock alert
            for signal in sentiment_data['sentiment_signals']:
                if signal['type'] == 'meme_stock_alert':
                    alerts.append(PortfolioAlert(
                        symbol=symbol,
                        alert_type='meme_stock_activity',
                        severity=AlertSeverity.WARNING,
                        message=signal['message'],
                        timestamp=datetime.now(),
                        data={'wsb_metrics': sentiment_data['reddit_metrics']['wsb_specific']},
                        action_required=True
                    ))
        
        return alerts
    
    def _generate_recommendations(self, risk_metrics: Dict, sentiment_data: Dict) -> List[Dict]:
        """
        Generate actionable recommendations based on analysis
        """
        recommendations = []
        
        if not risk_metrics:
            return recommendations
        
        # Position sizing recommendation
        if risk_metrics['risk_score'] > 70:
            recommendations.append({
                'type': 'reduce_position',
                'priority': 'high',
                'message': 'Consider reducing position size due to elevated risk',
                'rationale': f"Risk score of {risk_metrics['risk_score']:.0f} indicates high volatility ahead"
            })
        
        # Stop loss recommendation
        if risk_metrics['predicted_max_drawdown_5d'] < -5:
            stop_loss_price = risk_metrics['current_price'] * (1 + risk_metrics['predicted_max_drawdown_5d'] / 100 * 0.5)
            recommendations.append({
                'type': 'set_stop_loss',
                'priority': 'high',
                'message': f'Consider stop loss at ${stop_loss_price:.2f}',
                'rationale': f"Predicted drawdown of {abs(risk_metrics['predicted_max_drawdown_5d']):.1f}%"
            })
        
        # Sentiment-based recommendations
        if sentiment_data and sentiment_data['overall_sentiment']['score'] > 80:
            recommendations.append({
                'type': 'take_profits',
                'priority': 'medium',
                'message': 'Consider taking partial profits',
                'rationale': 'Extreme bullish sentiment often precedes pullbacks'
            })
        
        # Volatility hedging
        if risk_metrics['predicted_volatility_20d'] > 40:  # High volatility
            recommendations.append({
                'type': 'hedge_suggestion',
                'priority': 'medium',
                'message': 'Consider hedging with put options',
                'rationale': f"Volatility expected to remain elevated at {risk_metrics['predicted_volatility_20d']:.0f}%"
            })
        
        return recommendations
    
    def _calculate_diversification_score(self, portfolio_analysis: List[Dict]) -> float:
        """
        Calculate portfolio diversification score (0-100)
        """
        if len(portfolio_analysis) <= 1:
            return 0  # No diversification with single position
        
        # Extract risk scores
        risk_scores = []
        for analysis in portfolio_analysis:
            if analysis and 'combined_risk_score' in analysis:
                risk_scores.append(analysis['combined_risk_score']['score'])
        
        if not risk_scores:
            return 50  # Default
        
        # Calculate score based on:
        # 1. Number of positions (more is better up to a point)
        position_score = min(len(portfolio_analysis) * 10, 50)
        
        # 2. Risk distribution (want mix of risk levels)
        risk_std = np.std(risk_scores)
        risk_diversity_score = min(risk_std / 10, 50)
        
        return position_score + risk_diversity_score
    
    def _identify_trending_risks(self, portfolio_analysis: List[Dict]) -> List[Dict]:
        """
        Identify risks affecting multiple positions
        """
        trending_risks = []
        
        # Count risk types across portfolio
        volatility_spike_count = 0
        sentiment_extreme_count = 0
        meme_stock_count = 0
        
        for analysis in portfolio_analysis:
            if not analysis or 'alerts' not in analysis:
                continue
                
            for alert in analysis['alerts']:
                if alert.alert_type == 'volatility_spike':
                    volatility_spike_count += 1
                elif 'extreme' in alert.alert_type and 'sentiment' in alert.alert_type:
                    sentiment_extreme_count += 1
                elif alert.alert_type == 'meme_stock_activity':
                    meme_stock_count += 1
        
        # Market-wide volatility
        if volatility_spike_count >= 3:
            trending_risks.append({
                'type': 'market_volatility',
                'severity': 'high',
                'message': f'{volatility_spike_count} positions showing volatility spikes - potential market event',
                'affected_positions': volatility_spike_count
            })
        
        # Sentiment bubble
        if sentiment_extreme_count >= 2:
            trending_risks.append({
                'type': 'sentiment_bubble',
                'severity': 'medium',
                'message': 'Multiple positions showing extreme sentiment - market may be overheated',
                'affected_positions': sentiment_extreme_count
            })
        
        # Meme stock contagion
        if meme_stock_count >= 2:
            trending_risks.append({
                'type': 'meme_contagion',
                'severity': 'high',
                'message': 'Multiple meme stocks in portfolio - high correlation risk',
                'affected_positions': meme_stock_count
            })
        
        return trending_risks
    
    def _generate_action_items(self, portfolio_results: Dict) -> List[Dict]:
        """
        Generate prioritized action items for the user
        """
        action_items = []
        
        # Process critical alerts first
        critical_alerts = [a for a in portfolio_results['alerts'] 
                          if a.severity == AlertSeverity.CRITICAL and a.action_required]
        
        for alert in critical_alerts[:3]:  # Top 3 critical
            action_items.append({
                'priority': 1,
                'symbol': alert.symbol,
                'action': f"URGENT: {alert.message}",
                'type': alert.alert_type,
                'deadline': 'Immediate'
            })
        
        # Add trending risk actions
        for risk in portfolio_results['trending_risks']:
            if risk['severity'] == 'high':
                action_items.append({
                    'priority': 2,
                    'symbol': 'PORTFOLIO',
                    'action': risk['message'],
                    'type': risk['type'],
                    'deadline': 'Today'
                })
        
        # Add top recommendations
        for analysis in portfolio_results['portfolio_analysis']:
            if analysis and 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    if rec['priority'] == 'high':
                        action_items.append({
                            'priority': 3,
                            'symbol': analysis['symbol'],
                            'action': rec['message'],
                            'type': rec['type'],
                            'deadline': 'This week'
                        })
        
        # Sort by priority and limit
        action_items.sort(key=lambda x: x['priority'])
        return action_items[:5]  # Top 5 actions
    
    def get_market_pulse(self) -> Dict:
        """
        Get overall market sentiment and trending tickers
        """
        try:
            trending = self.sentiment_analyzer.get_trending_tickers()
            
            # Analyze top trending tickers
            market_sentiment = {
                'trending_tickers': trending,
                'market_mood': self._calculate_market_mood(trending),
                'timestamp': datetime.now().isoformat()
            }
            
            return market_sentiment
            
        except Exception as e:
            logger.error(f"Error getting market pulse: {e}")
            return {
                'trending_tickers': [],
                'market_mood': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_market_mood(self, trending_tickers: List[str]) -> str:
        """
        Calculate overall market mood from trending tickers
        """
        # This is simplified - in production you'd analyze sentiment of trending tickers
        # For now, just return a placeholder
        return "neutral"