"""
LLM-Powered Financial Insights

Transforms quantitative analysis results into human-readable investment insights.
Uses GPT-4 to explain:
- Regime dynamics and transitions
- Causal relationships between assets
- Strategy performance and risk metrics
- Investment recommendations and warnings
"""

import os
import json
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMInsightGenerator:
    """
    Generate narrative financial insights using GPT-4.
    
    Transforms quantitative analysis into investment-grade explanations.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize LLM insight generator
        
        Parameters:
        -----------
        model : str
            OpenAI model ('gpt-3.5-turbo' default, 'gpt-4' for higher quality if available)
        temperature : float
            Creativity level (0.0-1.0, default 0.7)
        """
        self.model = model
        self.temperature = temperature
        self.insights_cache = {}
    
    def _call_gpt(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call OpenAI GPT API
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        max_tokens : int
            Max response length
            
        Returns:
        --------
        str
            GPT response
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional quantitative analyst and portfolio manager. Explain complex financial analysis in clear, actionable terms for sophisticated investors."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ API Error: {str(e)[:200]}\n\nNote: Ensure OPENAI_API_KEY is set. Get it from https://platform.openai.com/api-keys"
    
    def explain_regime_characteristics(self, regime_stats: Dict) -> str:
        """
        Generate narrative explanation of market regimes
        
        Parameters:
        -----------
        regime_stats : dict
            Statistics for each regime from RegimeDetector
            
        Returns:
        --------
        str
            Human-readable regime explanation
        """
        prompt = f"""
Analyze these market regime statistics and explain them in investment terms:

Bull Regime:
- Frequency: {regime_stats.get('Bull', {}).get('percentage', 0):.1f}%
- Avg Return: {regime_stats.get('Bull', {}).get('avg_return', 0):.2%}
- Volatility: {regime_stats.get('Bull', {}).get('volatility', 0):.2%}

Neutral Regime:
- Frequency: {regime_stats.get('Neutral', {}).get('percentage', 0):.1f}%
- Avg Return: {regime_stats.get('Neutral', {}).get('avg_return', 0):.2%}
- Volatility: {regime_stats.get('Neutral', {}).get('volatility', 0):.2%}

Crisis Regime:
- Frequency: {regime_stats.get('Crisis', {}).get('percentage', 0):.1f}%
- Avg Return: {regime_stats.get('Crisis', {}).get('avg_return', 0):.2%}
- Volatility: {regime_stats.get('Crisis', {}).get('volatility', 0):.2%}

Provide:
1. Key characteristics of each regime
2. What triggers regime transitions
3. Historical patterns and cycles
4. Implications for portfolio construction
"""
        return self._call_gpt(prompt, max_tokens=1500)
    
    def explain_causal_relationships(self, gc_matrix: pd.DataFrame, var_results: Dict) -> str:
        """
        Explain Granger causality findings
        
        Parameters:
        -----------
        gc_matrix : pd.DataFrame
            Granger causality p-values
        var_results : dict
            VAR model statistics
            
        Returns:
        --------
        str
            Interpretation of causal relationships
        """
        # Extract significant relationships
        significant = []
        for cause in gc_matrix.index:
            for effect in gc_matrix.columns:
                p_val = gc_matrix.loc[cause, effect]
                if pd.notna(p_val) and p_val < 0.10 and cause != effect:
                    significant.append(f"{cause} → {effect} (p={p_val:.3f})")
        
        prompt = f"""
Interpret these causal relationships between financial assets:

Granger Causality Results (significant at p<0.10):
{', '.join(significant) if significant else 'No significant relationships found'}

Asset Correlation Matrix:
{var_results.get('correlation_matrix', 'Not provided')}

Explain:
1. Which assets lead/lag each other and why
2. Economic mechanisms behind these relationships
3. Implications for trading and hedging
4. Limitations of daily frequency analysis
5. When these relationships might break down (regime changes, market stress)
"""
        return self._call_gpt(prompt, max_tokens=1500)
    
    def explain_backtest_results(self, metrics_comparison: Dict) -> str:
        """
        Narrate backtest performance
        
        Parameters:
        -----------
        metrics_comparison : dict
            Strategy vs Benchmark metrics
            
        Returns:
        --------
        str
            Performance narrative with insights
        """
        strat = metrics_comparison.get('Strategy', {})
        bench = metrics_comparison.get('Benchmark', {})
        
        prompt = f"""
Analyze this backtest performance comparison and provide investment insights:

STRATEGY PERFORMANCE:
- Total Return: {strat.get('Total Return', 0):.2%}
- Annual Return: {strat.get('Annual Return', 0):.2%}
- Sharpe Ratio: {strat.get('Sharpe Ratio', 0):.2f}
- Max Drawdown: {strat.get('Max Drawdown', 0):.2%}
- Win Rate: {strat.get('Win Rate', 0):.1%}

BENCHMARK (Buy & Hold):
- Total Return: {bench.get('Total Return', 0):.2%}
- Annual Return: {bench.get('Annual Return', 0):.2%}
- Sharpe Ratio: {bench.get('Sharpe Ratio', 0):.2f}
- Max Drawdown: {bench.get('Max Drawdown', 0):.2%}
- Win Rate: {bench.get('Win Rate', 0):.1%}

Provide:
1. Key performance highlights and what drove them
2. Risk profile (volatility, drawdown) interpretation
3. Risk-adjusted return assessment
4. Strengths and weaknesses of the strategy
5. When the strategy underperformed and why
6. Realistic expectations for forward performance
7. Recommendations for portfolio managers
"""
        return self._call_gpt(prompt, max_tokens=2000)
    
    def generate_investment_thesis(self, 
                                   regime_analysis: str,
                                   causal_analysis: str,
                                   backtest_results: str) -> str:
        """
        Generate comprehensive investment thesis
        
        Parameters:
        -----------
        regime_analysis : str
            Regime explanation
        causal_analysis : str
            Causality explanation
        backtest_results : str
            Performance explanation
            
        Returns:
        --------
        str
            Unified investment thesis
        """
        prompt = f"""
You are writing an investment memo. Synthesize this analysis into a coherent thesis:

MARKET REGIME ANALYSIS:
{regime_analysis[:800]}...

ASSET RELATIONSHIPS (Causality):
{causal_analysis[:800]}...

STRATEGY PERFORMANCE:
{backtest_results[:800]}...

Create an executive summary that includes:
1. Market outlook and regime implications
2. Key portfolio drivers (what moves returns)
3. Risk factors and hedges
4. Tactical allocation recommendations
5. Expected return ranges and scenarios
6. Monitoring metrics and signals

Format as a professional investment memo (3-4 paragraphs).
"""
        return self._call_gpt(prompt, max_tokens=2000)
    
    def generate_risk_warnings(self, 
                              max_drawdown: float,
                              recent_volatility: float,
                              regime_transition_risk: float) -> str:
        """
        Highlight risks and warnings
        
        Parameters:
        -----------
        max_drawdown : float
            Maximum drawdown experienced
        recent_volatility : float
            Recent volatility level
        regime_transition_risk : float
            Probability of regime change
            
        Returns:
        --------
        str
            Risk assessment and warnings
        """
        prompt = f"""
Generate a risk assessment for a portfolio with these characteristics:
- Worst historical drawdown: {max_drawdown:.2%}
- Recent volatility: {recent_volatility:.2%} (annualized)
- Estimated regime transition prob: {regime_transition_risk:.1%}

Highlight:
1. Key tail risks and black swan scenarios
2. Correlation breakdowns during stress
3. Liquidity concerns
4. Model limitations and assumption failures
5. Stress test scenarios
6. Portfolio diversification gaps
7. Recommended hedges and risk controls

Format as risk disclosures a professional would present to investors.
"""
        return self._call_gpt(prompt, max_tokens=1500)
    
    def generate_quarterly_commentary(self,
                                     current_regime: str,
                                     regime_outlook: str,
                                     key_positions: List[str],
                                     recent_performance: Dict) -> str:
        """
        Generate quarterly investor commentary
        
        Parameters:
        -----------
        current_regime : str
            Current market regime
        regime_outlook : str
            Expected regime changes
        key_positions : list
            Current portfolio holdings
        recent_performance : dict
            YTD/QTD returns and metrics
            
        Returns:
        --------
        str
            Quarterly letter content
        """
        prompt = f"""
Write quarterly investor commentary (2-3 paragraphs):

Current Context:
- Market Regime: {current_regime}
- Regime Outlook: {regime_outlook}
- Key Positions: {', '.join(key_positions)}
- YTD Return: {recent_performance.get('ytd_return', 0):.2%}
- Volatility: {recent_performance.get('volatility', 0):.2%}
- Sharpe Ratio: {recent_performance.get('sharpe', 0):.2f}

Include:
1. Market environment summary
2. How your portfolio adapted to regime
3. Key successes and lessons learned
4. Forward outlook and strategy adjustments
5. Risk management actions taken
6. Expectations for next quarter

Tone: Professional, transparent, investor-focused
"""
        return self._call_gpt(prompt, max_tokens=1500)
    
    def summarize_key_insights(self, analysis_summary: Dict) -> str:
        """
        Generate bulleted key insights
        
        Parameters:
        -----------
        analysis_summary : dict
            Dictionary of all analysis results
            
        Returns:
        --------
        str
            Formatted bullet-point insights
        """
        prompt = f"""
Extract the most important investment insights from this analysis:

{json.dumps(analysis_summary, indent=2, default=str)[:2000]}

Format as:
🎯 KEY INSIGHTS (5-7 bullets)
- [Most important finding]
- [Risk factor]
- [Opportunity]
- [Portfolio implication]
- etc.

Then provide:
💡 ACTIONABLE RECOMMENDATIONS (3-5 items)
✚ Keep looking for:
- [What to monitor]
- [Warning signs]
"""
        return self._call_gpt(prompt, max_tokens=1000)
    
    def explain_in_plain_english(self, technical_result: str, context: str = "") -> str:
        """
        Translate technical/statistical results to plain English
        
        Parameters:
        -----------
        technical_result : str
            Complex statistical finding
        context : str
            Additional context
            
        Returns:
        --------
        str
            Simple explanation
        """
        prompt = f"""
Explain this technical finding in simple, plain English for a non-technical investor:

Technical Result:
{technical_result}

Context: {context}

Requirements:
- Avoid jargon (explain any necessary terms)
- Use real-world analogies
- Explain why it matters for investing
- Keep to 2-3 sentences max
"""
        return self._call_gpt(prompt, max_tokens=500)


class InsightGenerator:
    """High-level interface for generating all insights"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize insight generator"""
        self.llm = LLMInsightGenerator(model=model)
    
    def generate_full_report(self,
                            regime_stats: Dict,
                            gc_matrix: pd.DataFrame,
                            backtest_metrics: Dict,
                            var_stats: Dict = None) -> Dict[str, str]:
        """
        Generate complete analysis report with LLM insights
        
        Parameters:
        -----------
        regime_stats : dict
            Regime statistics
        gc_matrix : pd.DataFrame
            Granger causality matrix
        backtest_metrics : dict
            Backtest results
        var_stats : dict, optional
            VAR model statistics
            
        Returns:
        --------
        dict
            Complete report with all insights
        """
        print("🤖 Generating LLM insights...\n")
        
        print("📊 Analyzing regime characteristics...")
        regime_insights = self.llm.explain_regime_characteristics(regime_stats)
        
        print("🔗 Analyzing causal relationships...")
        causal_insights = self.llm.explain_causal_relationships(
            gc_matrix, 
            var_stats or {}
        )
        
        print("📈 Analyzing backtest performance...")
        performance_insights = self.llm.explain_backtest_results(backtest_metrics)
        
        print("⚠️  Assessing risks...")
        risk_warnings = self.llm.generate_risk_warnings(
            max_drawdown=backtest_metrics.get('Strategy', {}).get('Max Drawdown', -0.33),
            recent_volatility=backtest_metrics.get('Strategy', {}).get('Annual Volatility', 0.10),
            regime_transition_risk=0.15
        )
        
        print("💡 Generating investment thesis...")
        thesis = self.llm.generate_investment_thesis(
            regime_insights,
            causal_insights,
            performance_insights
        )
        
        print("✅ Complete!\n")
        
        return {
            'Regime Analysis': regime_insights,
            'Causal Relationships': causal_insights,
            'Performance Insights': performance_insights,
            'Risk Assessment': risk_warnings,
            'Investment Thesis': thesis
        }
    
    def print_report(self, report: Dict[str, str]) -> None:
        """Pretty print full report"""
        print("\n" + "="*80)
        print("📋 QUANTITATIVE ANALYSIS REPORT - LLM INSIGHTS")
        print("="*80)
        
        for section, content in report.items():
            print(f"\n{'─'*80}")
            print(f"📌 {section.upper()}")
            print(f"{'─'*80}\n")
            print(content)
        
        print("\n" + "="*80)
