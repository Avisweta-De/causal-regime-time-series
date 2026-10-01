"""
Grounded Insight Reports (optional LLM narration)

The analysis numbers always come from code. This module:
1. Collects computed results into a plain `facts` dictionary
2. Renders a deterministic Markdown report from those facts (no API needed)
3. Optionally asks an LLM to explain the facts in plain English, with
   instructions to use only the supplied numbers, and then flags any
   number in the LLM text that does not appear in the facts

Earlier versions printed hand-written "mock" LLM output containing figures
that were never computed. That is why the report is now built from facts
first and the LLM is only a narrator.
"""

import os
import re
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

GROUNDING_RULES = (
    "Use ONLY the numbers in FACTS. Do not introduce any other statistics, "
    "dates, prices, forecasts or market commentary. If something is not in "
    "FACTS, say it was not analysed. Do not describe results as guaranteed or "
    "as investment advice. Be concise and plain."
)


def _pct(x, digits=1):
    return None if x is None or pd.isna(x) else round(float(x) * 100, digits)


def build_facts(regime_characteristics: pd.DataFrame,
                transition_matrix: pd.DataFrame,
                walk_forward: Dict[str, Dict],
                robustness: Optional[pd.DataFrame] = None,
                significant_granger: Optional[pd.DataFrame] = None,
                correlations: Optional[Dict[str, float]] = None,
                period: Optional[str] = None,
                settings: Optional[Dict] = None) -> Dict:
    """
    Collect computed results into a JSON-serialisable dict.

    Parameters:
    -----------
    regime_characteristics : DataFrame from RegimeDetector.get_regime_characteristics()
    transition_matrix : DataFrame from RegimeDetector.analyze_regime_transitions()['transition_matrix']
    walk_forward : {'Strategy': metrics, 'Benchmark': metrics, optional 'Constant mix': metrics}
        from performance_metrics(); 'Constant mix' holds the strategy's average equity weight
    robustness : DataFrame from backtesting.robustness_check(), optional
    significant_granger : DataFrame from CausalityAnalyzer.get_significant_causality(), optional
    correlations : {pair_name: correlation}, optional
    period : str, optional, e.g. '2010-02-02 to 2024-12-30'
    settings : dict, optional
        Backtest settings, e.g. {'refit': 'monthly', 'train_window_days': 504,
        'cost_bps': 5, 'signal_lag_days': 1, 'granger_tests': 20}
    """
    regimes = {}
    for name, row in regime_characteristics.iterrows():
        regimes[name] = {
            'share_of_days_pct': round(float(row['% of Days']), 1),
            'daily_volatility_next_day_pct': _pct(row['Volatility (next day)'], 2),
            'mean_daily_return_next_day_pct': _pct(row['Mean Return (next day)'], 3),
        }

    stay = {name: _pct(transition_matrix.loc[name, name]) for name in transition_matrix.index}

    def pack(m):
        return {
            'cagr_pct': _pct(m['Annual Return']),
            'annual_volatility_pct': _pct(m['Annual Volatility']),
            'sharpe': round(float(m['Sharpe Ratio']), 2),
            'max_drawdown_pct': _pct(m['Max Drawdown']),
            'start': str(pd.Timestamp(m['Start']).date()),
            'end': str(pd.Timestamp(m['End']).date()),
        }

    facts = {
        'period': period,
        'settings': settings or {},
        'regimes': regimes,
        'probability_regime_continues_next_day_pct': stay,
        'walk_forward_strategy': pack(walk_forward['Strategy']),
        'walk_forward_buy_and_hold': pack(walk_forward['Benchmark']),
    }
    if 'Constant mix' in walk_forward:
        facts['walk_forward_constant_mix'] = pack(walk_forward['Constant mix'])
        if 'equity_weight' in walk_forward:
            facts['walk_forward_constant_mix']['equity_weight_pct'] = _pct(walk_forward['equity_weight'], 0)

    if robustness is not None and len(robustness):
        diff_cagr = (robustness['strategy_cagr'] - robustness['benchmark_cagr']) * 100
        dd_cut = (robustness['strategy_max_dd'] - robustness['benchmark_max_dd']) * 100
        facts['robustness'] = {
            'runs': int(len(robustness)),
            'runs_with_higher_sharpe_than_buy_and_hold': int((robustness['strategy_sharpe'] > robustness['benchmark_sharpe']).sum()),
            'runs_with_smaller_max_drawdown': int((robustness['strategy_max_dd'] > robustness['benchmark_max_dd']).sum()),
            'cagr_difference_pct_median': round(float(diff_cagr.median()), 1),
            'cagr_difference_pct_range': [round(float(diff_cagr.min()), 1), round(float(diff_cagr.max()), 1)],
            'max_drawdown_reduction_pct_points_median': round(float(dd_cut.median()), 1),
            'max_drawdown_reduction_pct_points_range': [round(float(dd_cut.min()), 1), round(float(dd_cut.max()), 1)],
        }
        if 'constant_mix_sharpe' in robustness:
            facts['robustness'].update({
                'runs_with_smaller_max_drawdown_than_constant_mix': int((robustness['strategy_max_dd'] > robustness['constant_mix_max_dd']).sum()),
                'runs_with_higher_sharpe_than_constant_mix': int((robustness['strategy_sharpe'] > robustness['constant_mix_sharpe']).sum()),
                'runs_with_higher_cagr_than_constant_mix': int((robustness['strategy_cagr'] > robustness['constant_mix_cagr']).sum()),
            })

    if significant_granger is not None:
        facts['granger_significant_after_bonferroni'] = [
            {'cause': r.cause, 'effect': r.effect, 'p_value_adjusted': float(f'{r.p_value_adjusted:.2g}')}
            for r in significant_granger.itertuples()
        ]
    if correlations:
        facts['correlations'] = {k: round(float(v), 2) for k, v in correlations.items()}
    return facts


def _settings_sentence(st: Dict) -> str:
    parts = ["Model refit on past data only"]
    if st.get('refit'):
        parts[0] += f" ({st['refit']}"
        parts[0] += f", {st['train_window_days']}-day training window)" if st.get('train_window_days') else ")"
    if st.get('signal_lag_days'):
        parts.append(f"signals applied after {st['signal_lag_days']} day(s)")
    if st.get('cost_bps') is not None:
        bps = st['cost_bps']
        bps = int(bps) if float(bps).is_integer() else bps
        parts.append(f"{bps} bps trading cost per unit of turnover")
    return ", ".join(parts) + "."


def render_report(facts: Dict) -> str:
    """Deterministic Markdown report built only from `facts`."""
    s, b = facts['walk_forward_strategy'], facts['walk_forward_buy_and_hold']
    lines = ["# Regime Risk Analysis: Summary Report", ""]
    if facts.get('period'):
        lines += [f"Data: S&P 500 and four cross-asset series, {facts['period']}.", ""]

    lines += ["## Regimes (volatility states of the S&P 500)", "",
              "| Regime | Share of days | Next-day daily volatility | Next-day mean return | P(still in regime tomorrow) |",
              "|---|---|---|---|---|"]
    for name, r in facts['regimes'].items():
        stay = facts['probability_regime_continues_next_day_pct'].get(name)
        lines.append(f"| {name} | {r['share_of_days_pct']}% | {r['daily_volatility_next_day_pct']}% | "
                     f"{r['mean_daily_return_next_day_pct']}% | {stay}% |")
    vols = [r['daily_volatility_next_day_pct'] for r in facts['regimes'].values()]
    lines += [""]
    if all(a < b for a, b in zip(vols, vols[1:])):
        lines += ["Next-day volatility rises from regime to regime, so the regimes carry real information about near-term risk. "
                  "Next-day mean returns are small and noisy by comparison, so the regimes say little about direction.", ""]

    lines += ["## Out-of-sample (walk-forward) backtest", "",
              _settings_sentence(facts.get('settings', {})) + f" Period: {s['start']} to {s['end']}.", "",
              ]
    c = facts.get('walk_forward_constant_mix')
    if c:
        w = c.get('equity_weight_pct')
        mix = f"Fixed {w:.0f}% equity mix" if w is not None else "Fixed equity mix"
        lines += [f"| | Regime strategy | {mix} | Buy & hold |", "|---|---|---|---|",
                  f"| CAGR | {s['cagr_pct']}% | {c['cagr_pct']}% | {b['cagr_pct']}% |",
                  f"| Annual volatility | {s['annual_volatility_pct']}% | {c['annual_volatility_pct']}% | {b['annual_volatility_pct']}% |",
                  f"| Sharpe ratio | {s['sharpe']} | {c['sharpe']} | {b['sharpe']} |",
                  f"| Max drawdown | {s['max_drawdown_pct']}% | {c['max_drawdown_pct']}% | {b['max_drawdown_pct']}% |", ""]
        if w is not None:
            lines += [f"The fixed mix holds the strategy's average equity weight ({w:.0f}%) every day. It shows how much of the "
                      "risk reduction comes from timing rather than simply from holding less equity.", ""]
    else:
        lines += ["| | Regime strategy | Buy & hold |", "|---|---|---|",
                  f"| CAGR | {s['cagr_pct']}% | {b['cagr_pct']}% |",
                  f"| Annual volatility | {s['annual_volatility_pct']}% | {b['annual_volatility_pct']}% |",
                  f"| Sharpe ratio | {s['sharpe']} | {b['sharpe']} |",
                  f"| Max drawdown | {s['max_drawdown_pct']}% | {b['max_drawdown_pct']}% |", ""]

    rb = facts.get('robustness')
    if rb:
        lines += ["## Robustness", "",
                  f"Across {rb['runs']} runs (different random seeds and training windows): "
                  f"the strategy had a smaller max drawdown in {rb['runs_with_smaller_max_drawdown']} runs and a higher Sharpe ratio in "
                  f"{rb['runs_with_higher_sharpe_than_buy_and_hold']}. CAGR minus buy-and-hold had a median of "
                  f"{rb['cagr_difference_pct_median']} points (range {rb['cagr_difference_pct_range'][0]} to {rb['cagr_difference_pct_range'][1]}). "
                  f"Max drawdown was reduced by a median of {rb['max_drawdown_reduction_pct_points_median']} points "
                  f"(range {rb['max_drawdown_reduction_pct_points_range'][0]} to {rb['max_drawdown_reduction_pct_points_range'][1]}).", ""]
        if 'runs_with_higher_sharpe_than_constant_mix' in rb:
            lines += [f"Against the fixed mix with the same average equity weight, the strategy had a smaller max drawdown in "
                      f"{rb['runs_with_smaller_max_drawdown_than_constant_mix']} of {rb['runs']} runs, a higher Sharpe ratio in "
                      f"{rb['runs_with_higher_sharpe_than_constant_mix']} and a higher CAGR in {rb['runs_with_higher_cagr_than_constant_mix']}.", ""]

    if 'granger_significant_after_bonferroni' in facts:
        sig = facts['granger_significant_after_bonferroni']
        lines += ["## Lead-lag (Granger) tests", ""]
        if sig:
            n_tests = facts.get('settings', {}).get('granger_tests', 'all')
            lines += [f"Pairs where yesterday's return of one asset helps predict today's return of another, after Bonferroni adjustment for {n_tests} tests:", ""]
            lines += [f"- {g['cause']} → {g['effect']} (adjusted p = {g['p_value_adjusted']})" for g in sig]
        else:
            lines += ["No pair is significant after adjusting for multiple tests."]
        lines += ["", "Granger tests measure predictability, not economic causation. Assets that close at different times of day can show lead-lag effects mechanically, and this analysis did not test whether these effects are large enough to trade after costs.", ""]

    if facts.get('correlations'):
        lines += ["## Correlations of daily returns", ""]
        lines += [f"- {k}: {v}" for k, v in facts['correlations'].items()]
        lines += [""]

    lines += ["## Limitations", "",
              "- One equity index and one historical period; results may not hold in other markets or periods.",
              "- Cash is assumed to earn 0%, and taxes, slippage and market impact are not modelled.",
              "- The regime model is a statistical description, not a forecast of crashes.", ""]
    return "\n".join(lines)


def _numbers_in(text: str) -> List[float]:
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))]


def ungrounded_numbers(text: str, facts: Dict, tolerance: float = 0.051) -> List[float]:
    """
    Numbers in `text` that do not match any number in `facts` (within tolerance).

    Small integers (0-10) are ignored because they are usually list markers or
    counts in ordinary prose.
    """
    known = _numbers_in(json.dumps(facts))
    known += [abs(k) for k in known]
    flagged = []
    for n in _numbers_in(text):
        if abs(n) <= 10 and float(n).is_integer():
            continue
        if not any(abs(abs(n) - abs(k)) <= tolerance for k in known):
            flagged.append(n)
    return flagged


class LLMInsightGenerator:
    """Optional plain-English narration of computed facts using the OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

    @staticmethod
    def is_configured() -> bool:
        """True when an OpenAI key is available in the environment."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def _call_gpt(self, prompt: str, max_tokens: int = 800) -> str:
        from openai import OpenAI  # imported lazily: the package is optional
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You explain quantitative risk analysis to non-technical readers. " + GROUNDING_RULES},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def narrate(self, facts: Dict, audience: str = "a non-technical risk committee",
                max_tokens: int = 600) -> Dict[str, object]:
        """
        Plain-English summary of `facts` for the given audience.

        Returns:
        --------
        dict with 'text' and 'ungrounded_numbers' (numbers in the text that
        are not in the facts; review these before using the text)
        """
        prompt = (f"FACTS (JSON):\n{json.dumps(facts, indent=2)}\n\n"
                  f"Write 2 short paragraphs for {audience}: what the regimes show about risk, "
                  f"and what the out-of-sample backtest and robustness results do and do not show.")
        text = self._call_gpt(prompt, max_tokens=max_tokens)
        return {'text': text, 'ungrounded_numbers': ungrounded_numbers(text, facts)}

    def explain_in_plain_english(self, technical_result: str, context: str = "") -> str:
        """Explain one technical finding in 2-3 plain sentences."""
        prompt = (f"FACTS:\n{technical_result}\n\nContext: {context}\n\n"
                  "Explain this in 2-3 plain sentences and say why it matters for managing risk.")
        return self._call_gpt(prompt, max_tokens=300)


class InsightGenerator:
    """Build the report from facts, optionally adding an LLM narration."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLMInsightGenerator(model=model)

    def generate_full_report(self, facts: Dict, use_llm: Optional[bool] = None) -> Dict[str, object]:
        """
        Parameters:
        -----------
        facts : dict from build_facts()
        use_llm : bool, optional
            Default: use the LLM only if OPENAI_API_KEY is set

        Returns:
        --------
        dict with 'markdown' (deterministic report), 'llm_summary' (or None)
        and 'ungrounded_numbers'
        """
        report = render_report(facts)
        use_llm = self.llm.is_configured() if use_llm is None else use_llm
        summary, flagged = None, []
        if use_llm:
            out = self.llm.narrate(facts)
            summary, flagged = out['text'], out['ungrounded_numbers']
            report += ("\n## Plain-English summary (LLM-written from the facts above)\n\n" + summary + "\n")
            if flagged:
                report += f"\n> Check before use: these numbers are not in the computed facts: {flagged}\n"
        return {'markdown': report, 'llm_summary': summary, 'ungrounded_numbers': flagged}
