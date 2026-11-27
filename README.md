# BIST Backtest

[![tests](https://github.com/kaan-svg/bist-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/kaan-svg/bist-backtest/actions/workflows/tests.yml)

A backtesting engine for trading strategies on Borsa Istanbul (BIST) stocks, built step by step with tests.
Strategies are run on historical data with realistic costs and compared against simply holding the
stock, the BIST 100 index and a deposit rate.

The goal is not to find a strategy that "beats the market". It is to measure honestly, and to show
where backtests usually go wrong.

> ⚠️ **For research and learning only.** This is not investment advice and does not trade real money.
> Past performance does not predict future results, and backtests tend to overstate them
> (see *Pitfalls found so far*).

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest

python -m src.veri THYAO.IS 2018-01-01 2025-12-31       # download + clean data
python -m src.olcutler THYAO.IS                          # strategies with performance metrics
python -m src.karsilastirma THYAO.IS --faiz 0.30         # vs BIST 100 and a deposit rate
```

Data comes from Yahoo Finance (BIST symbols end in `.IS`) and is cached in `data/`.
Tests use synthetic data and need no internet.

## How it works

```
veri.py          download, cache and clean prices
strateji.py      strategies: price data -> daily target position (0 = cash, 1 = stock)
motor.py         signals -> trades at next day's open -> daily returns, costs, equity curve
olcutler.py      annual return, volatility, Sharpe, max drawdown, trade statistics
karsilastirma.py comparison with BIST 100 buy-and-hold and a deposit rate
```

| Strategy | Rule |
|---|---|
| `al_tut` | Buy on the first day, hold to the end. The baseline every strategy must beat. |
| `hareketli_ortalama(kisa, uzun)` | Hold the stock while the short moving average is above the long one. |

**Costs** are applied on every buy and every sell: commission (default 0.15%, a mid-range retail
rate), BSMV tax (5% of commission) and slippage (0.05%, an assumption). About 0.21% per trade in total;
all three are parameters.

## Pitfalls found so far

Each of these would have made the results look better than they are. Each is covered by a test.

1. **Trading at the signal day's close.** A signal computed from today's close cannot be traded at
   today's close: that price is only known after the market closes. The engine trades at the next
   day's open. A test checks that the gain between the two prices is *not* counted.
2. **Holiday rows in the data.** Yahoo returns rows for days the exchange was closed (bayram,
   New Year) with zero volume and a flat price: 22 such days for THYAO in 2018–2025. Left in, they
   count as real trading days with zero return, which lowers volatility and raises the Sharpe ratio,
   and lets a strategy "trade" on a closed day.
3. **A lookahead test that did not catch lookahead.** Every strategy is tested by changing future prices
   and checking that past signals stay the same. A deliberately cheating strategy (`shift(-1)`)
   *passed* the first version of that test, because it changed the future at a single point by a random
   amount. The test now changes the future at many points, both up and down, and the cheating strategy
   is caught. The test for the test is part of the suite.
4. **Mixing time scales.** The deposit-rate curve grew per trading day (252/year) while annual return was
   measured in calendar years, so a 30% rate looked like 31.2%. Both now use calendar time.
5. **The BIST 100 redenomination.** Two zeros were removed from the index on 27 July 2020. If the data
   source has not adjusted for it, the index appears to fall 99% in one day. One-day jumps larger than
   20× are treated as scale changes and earlier prices are rescaled.
6. **Nominal TL returns.** With deposit rates between roughly 8% and 50% over the period, a 40% annual
   return in TL can be no better than a bank deposit. Sharpe is computed with a zero risk-free rate by
   default, which flatters every strategy; the comparison with a deposit rate is reported separately.

## Results

<!-- RESULTS -->
_To be added from a run on real data._

## Roadmap

- [x] Data layer: download, cache, cleaning (holiday rows)
- [x] Strategy interface with automatic lookahead tests
- [x] Engine: next-day-open execution, costs
- [x] Metrics and comparison with BIST 100 and a deposit rate
- [ ] RSI and momentum strategies
- [ ] Parameter search
- [ ] Overfitting check: train/test split, walk-forward
- [ ] Multiple stocks and a portfolio
- [ ] Charts: equity curve, drawdown
