# BIST Backtest

A backtesting engine for trading strategies on Borsa Istanbul (BIST) stocks, built step by step
with tests. Strategies are evaluated on historical data with realistic costs and compared against
simply holding the stock and the BIST 100 index.

> ⚠️ **For research and learning only.** This is not investment advice and does not trade real money.
> A strategy that performed well in the past is not expected to perform well in the future;
> backtests in particular tend to overstate results (see *Pitfalls* below, once written).

**Status:** work in progress.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.veri THYAO.IS 2018-01-01 2025-12-31
```
