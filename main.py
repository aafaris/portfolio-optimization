import copy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, plotting
from pypfopt import expected_returns
from pypfopt import risk_models
import asyncio
import aiohttp

# perpetual futures contracts for CQ - Current Quarter
# CW - Current Week
# NW - Next Week
# CQ - Current Quarter
# NQ - Next Quarter
tickers = ["BTC_CQ", "ETH_CQ", "LTC_CQ"]
start_datetime = "2022-11-01T00:00:00+00:00"
end_datetime = "2022-12-01T23:00:00+00:00"
period = "60min"


def convert_datetime_to_timestamp(dt):
    date = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S%z")
    return int(date.timestamp())


def get_tasks(start, end, session):
    tasks = []
    for ticker in tickers:
        url = f"https://api.hbdm.com/market/history/kline?symbol={ticker}&period={period}&from={start}&to={end}"
        tasks.append(session.get(url, ssl=False))

    return tasks


async def get_market_data(start, end):
    assets = []
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(start, end, session)
        responses = await asyncio.gather(*tasks)
        for i, res in enumerate(responses):

            data = await res.json()

            if data["status"] == "ok":
                df = pd.DataFrame(data["data"])
                df = df[["id", "close"]].set_index('id').rename(columns={"close": tickers[i]})
                assets.append(df)
            else:
                error_ticker = tickers.pop(i)
                print(f"Error retrieving {error_ticker}:", data["err-code"])

    # merging all assets prices
    all_assets = pd.concat(assets, axis=1)

    return all_assets[tickers]


def get_mu_and_sigma():

    start = convert_datetime_to_timestamp(start_datetime)
    end = convert_datetime_to_timestamp(end_datetime)

    # get perpetual futures contracts
    all_assets = asyncio.get_event_loop().run_until_complete(get_market_data(start, end))

    # 365 trading days per year
    mu = expected_returns.ema_historical_return(all_assets, frequency=365)
    sigma = risk_models.exp_cov(all_assets, frequency=365)

    return mu, sigma


def plot_efficient_frontier(mu, sigma):

    fig, ax = plt.subplots()
    ef = EfficientFrontier(mu, sigma, weight_bounds=(-1, 1))

    # plot max sharpe ratio
    ef_max_sharpe = copy.copy(ef)
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    ax.set_title("Mean Variance Efficient Frontier")
    # plot ticker symbol
    for i, text in enumerate(ef.tickers):
        ax.annotate(text, ((np.diag(ef.cov_matrix) ** (1/2))[i], ef.expected_returns[i]))

    plotting.plot_efficient_frontier(ef, ax=ax)
    plt.show()


def calculate_mean_variance(mu, sigma):

    # by default, risk-free rate of 0.02, allow shorting with -1 weight bounds
    ef = EfficientFrontier(mu, sigma, weight_bounds=(-1, 1))
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print("Output: ", dict(cleaned_weights))

    ef.portfolio_performance(verbose=True)


if __name__ == '__main__':
    mu, S = get_mu_and_sigma()
    calculate_mean_variance(mu, S)
    plot_efficient_frontier(mu, S)
