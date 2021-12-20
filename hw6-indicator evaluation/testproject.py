import TheoreticallyOptimalStrategy as tos
import datetime as dt
import pandas as pd
import numpy as np
import indicators


def author():
    return "sgorucu3"


if __name__ == "__main__":
    # input = dict(
    #     symbol="AAPL",
    #     sd=dt.datetime(2010, 1, 1),
    #     ed=dt.datetime(2011, 12, 31),
    #     sv=100000,
    # )

    input = dict(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    )

    tos.testPolicy(**input)

    input = dict(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
    )
    indicators.indicator(**input)

