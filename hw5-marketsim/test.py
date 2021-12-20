import marketsim
import numpy as np
import pandas as pd


def get_stats(port_val):
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * daily_rets.mean() / std_daily_ret
    return avg_daily_ret, sharpe_ratio


# ## 1
# inputs = dict(
#     orders_file="orders/orders-01.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
#
# outputs = dict(
#     num_days=245,
#     last_day_portval=1115569.2,
#     sharpe_ratio=0.612340613407,
#     avg_daily_ret=0.00055037432146,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
#
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 2
# inputs = dict(
#     orders_file="orders/orders-02.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs = dict(
#     num_days=245,
#     last_day_portval=1095003.35,
#     sharpe_ratio=1.01613520942,
#     avg_daily_ret=0.000390534819609,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
#
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 3
# inputs = dict(
#     orders_file="orders/orders-03.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs = dict(
#     num_days=240,
#     last_day_portval=857616.0,
#     sharpe_ratio=-0.759896272199,
#     avg_daily_ret=-0.000571326189931,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 4
# inputs=dict(
#     orders_file="orders/orders-04.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs=dict(
#     num_days=233,
#     last_day_portval=923545.4,
#     sharpe_ratio=-0.266030146916,
#     avg_daily_ret=-0.000240200768212,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 5
# inputs=dict(
#     orders_file="orders/orders-05.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs=dict(
#     num_days=296,
#     last_day_portval=1415563.0,
#     sharpe_ratio=2.19591520826,
#     avg_daily_ret=0.00121733290744,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 6
# inputs=dict(
#     orders_file="orders/orders-06.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs=dict(
#     num_days=210,
#     last_day_portval=894604.3,
#     sharpe_ratio=-1.23463930987,
#     avg_daily_ret=-0.000511281541086,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 7
# inputs=dict(
#     orders_file="orders/orders-07.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs=dict(
#     num_days=237,
#     last_day_portval=1106563.3,
#     sharpe_ratio=2.10356512897,
#     avg_daily_ret=0.0004345040621,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)
#
# ## 8
# inputs=dict(
#     orders_file="orders/orders-08.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs=dict(
#     num_days=229,
#     last_day_portval=1074884.1,
#     sharpe_ratio=0.941858298061,
#     avg_daily_ret=0.000332404156893,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)

## 9
# inputs = dict(
#     orders_file="orders/orders-09.csv",
#     start_val=1000000,
#     commission=0.0,
#     impact=0.0,
# )
# outputs = dict(
#     num_days=37,
#     last_day_portval=1067710.0,
#     sharpe_ratio=2.90848480553,
#     avg_daily_ret=0.00187252252117,
# )
#
# portvals = marketsim.compute_portvals(**inputs)
# avg_daily_ret, sharpe_ratio = get_stats(portvals)


inputs = dict(
    orders_file="additional_orders/orders-short.csv",
    start_val=1000000,
)
portvals = marketsim.compute_portvals(**inputs)
avg_daily_ret, sharpe_ratio = get_stats(portvals)

inputs = dict(
    orders_file="additional_orders/orders2.csv",
    start_val=1000000,
)
portvals = marketsim.compute_portvals(**inputs)
avg_daily_ret, sharpe_ratio = get_stats(portvals)

pass
