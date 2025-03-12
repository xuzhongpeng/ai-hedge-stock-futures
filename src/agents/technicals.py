import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import get_prices, prices_to_df
from utils.progress import progress


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    assets = data["assets"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    realtime = state["metadata"]["realtime"]

    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            realtime=realtime,
        )

        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        progress.update_status("technical_analyst_agent", ticker, "Calculating super trend signals")
        super_trend_signals = calculate_super_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "super_trend": 0.3,
            "mean_reversion": 0.2,
            "momentum": 0.2,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "super_trend": super_trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "super_trend": {
                    "signal": super_trend_signals["signal"],
                    "confidence": round(super_trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(super_trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name="technical_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def has_long_monotonic_decreasing_segment(lst, cnt):
    if len(lst) < 300:
        return 0

    lst = list(lst)
    l = len(lst)

    # 初始化变量
    idx = 0
    in_descending_segment = False
    count = 1  # 初始设置为1而非0，因单个元素也可视作“区间”
    max_count = 0  # 用于记录最长下降区间的元素数量

    for i in range(l - 299, l):
        if l - idx > 24:
            idx = 0

        # 标记下降区间
        if lst[i-1] > lst[i]:
            count += 1
            in_descending_segment = True
            idx = i
        else:  # 当前元素不小于前面的元素，即不满足单调下降
            if in_descending_segment and count > max_count:
                # 更新最长下降区间计数
                max_count = count
                idx = i

                if max_count >= cnt and l - idx <= 10:
                    # print(f"total seg={l}, idx={idx}, max decreasing seg={max_count}")
                    return max_count

            in_descending_segment = False
            count = 1   # 重新开始统计新段落的长度

    return count

def has_long_monotonic_increasing_segment(lst, cnt):
    if len(lst) < 300:
        return 0

    lst = list(lst)
    l = len(lst)

    # 初始化变量
    idx = 0
    in_inscending_segment = False
    count = 1  # 初始设置为1而非0，因单个元素也可视作“区间”
    max_count = 0  # 用于记录最长下降区间的元素数量

    for i in range(l - 299, l):
        if l - idx > 24:
            idx = 0

        # 标记上升区间
        if lst[i-1] < lst[i]:
            count += 1
            in_inscending_segment = True
            idx = i
        else:  # 当前元素不小于前面的元素，即不满足单调下降
            if in_inscending_segment and count > max_count:
                # 更新最长下降区间计数
                max_count = count
                idx = i

                if max_count >= cnt and l - idx <= 10:
                    # print(f"total seg={l}, idx={idx}, max decreasing seg={max_count}")
                    return max_count

            in_inscending_segment = False
            count = 1   # 重新开始统计新段落的长度

    return count


def get_supertrend(high, low, close, lookback, multiplier):
    # ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands[final_bands.columns[1]] = final_bands[final_bands.columns[0]]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band.iloc[i] < final_bands.iloc[i-1,0]) | (close.iloc[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band.iloc[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]

    # FINAL LOWER BAND
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band.iloc[i] > final_bands.iloc[i-1,1]) | (close.iloc[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band.iloc[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]

    # SUPERTREND
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper']]
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close.iloc[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close.iloc[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close.iloc[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close.iloc[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    #supertrend = supertrend.dropna()[1:]
    supertrend.iloc[0] = supertrend.iloc[1]

    # ST UPTREND/DOWNTREND
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close.iloc[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close.iloc[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    return st, upt, dt


# 计算滚动窗口内的斜率
def calculate_slope(x):
    # 将 NaN 转换为 np.nan
    x = np.array(x, dtype=np.float64)
    # 检查窗口中的 NaN 值数量
    if np.isnan(x).sum() > 0:
        return np.nan
    # 计算拟合直线的斜率
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope


def calculate_super_trend_signals(prices_df):
    """
    Advanced super trend following strategy using multiple timeframes and indicators
    """
    st, upt, dt = get_supertrend(prices_df['high'], prices_df['low'], prices_df['close'], 11, 3)

    prices_df['MA34'] = prices_df['close'].rolling(window=34).mean()
    prices_df['MA55'] = prices_df['close'].rolling(window=55).mean()
    prices_df['MA127'] = prices_df['close'].rolling(window=127).mean()
    prices_df['slope_of_MA34'] = prices_df['MA34'].rolling(window=3).apply(calculate_slope, raw=True).round(3)
    prices_df['slope_of_MA55'] = prices_df['MA55'].rolling(window=3).apply(calculate_slope, raw=True).round(3)
    prices_df['slope_of_MA127'] = prices_df['MA127'].rolling(window=3).apply(calculate_slope, raw=True).round(3)

    total_length = len(prices_df)
    golden_candidate = 0
    bull = False
    if total_length > 60:
        for i in range(len(prices_df)):
            if i < 300:
                continue

            if has_long_monotonic_decreasing_segment(prices_df['MA34'].iloc[:i], 120) > 120 or has_long_monotonic_decreasing_segment(prices_df['MA55'].iloc[:i], 120) > 120:
                golden_candidate = 200

            if golden_candidate:
                golden_candidate -= 1

    if (prices_df['slope_of_MA55'].iloc[-1] > prices_df['slope_of_MA55'].iloc[-2] > prices_df['slope_of_MA55'].iloc[-3] > prices_df['slope_of_MA55'].iloc[-4] or \
        prices_df['slope_of_MA34'].iloc[-1] > prices_df['slope_of_MA34'].iloc[-2] > prices_df['slope_of_MA34'].iloc[-3] > prices_df['slope_of_MA34'].iloc[-4] or \
        prices_df.tail(5)['slope_of_MA34'].min() > -0.4) and \
        prices_df.tail(15)['slope_of_MA34'].abs().mean() > 0.25 and \
        prices_df.iloc[-15:].apply(lambda row: row['close'] > row['MA34'], axis=1).sum() >= 5 and \
        prices_df.iloc[-15:].apply(lambda row: row['close'] < row['MA55'], axis=1).sum() > 10 and \
        golden_candidate:
            bull = True

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if prices_df['close'].iloc[-1] > st.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    else:
        signal = "bearish" if not bull else "bullish"
        confidence = trend_strength

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
            "bull": bull
        },
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(55).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = 1 / (1 + np.exp(-abs(momentum_score) * 10))
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = 1 / (1 + np.exp(-abs(momentum_score) * 10))
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_hurst_exponent(price_series, max_lag=20):
    """修正后的Hurst指数计算（强制约束0-1）"""
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        price_detrended = price_series - np.polyval(np.polyfit(np.arange(len(price_series)), price_series, 1),
                                                    np.arange(len(price_series)))
        tau.append(np.sqrt(np.std(price_detrended[lag:] - price_detrended[:-lag])))
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2
    return max(0.0, min(1.0, hurst))  # 约束在0-1之间


def calculate_stat_arb_signals(prices_df):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    # hurst = calculate_hurst_exponent(prices_df["close"])

    # 滚动计算Hurst指数（示例窗口63天）
    hurst = prices_df["close"].rolling(63).apply(
        lambda x: calculate_hurst_exponent(x), raw=True
    )
    hurst = hurst.iloc[-1]  # 取最新值

    # 基于Hurst指数的置信度（0到1）
    hurst_confidence = max(0.0, min(1.0, (0.5 - hurst) * 2))

    # 偏度调整（绝对值越大，置信度越高）
    skew_factor = 1 - 1 / (1 + np.exp(abs(skew.iloc[-1]) - 1))  # Sigmoid函数

    # 峰度调整（接近正态分布时置信度更高）
    kurt_penalty = 1 / (1 + np.exp(kurt.iloc[-1] - 3))  # 对比正态分布峰度3

    confidence = np.clip(hurst_confidence * skew_factor * kurt_penalty, 0, 1)

    # Correlation analysis
    # (would include correlation with related securities in real implementation)

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5
