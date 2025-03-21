from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.api import get_insider_trades, get_company_news
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.llm import call_llm


class SentimentDecision(BaseModel):
    sentiment: Literal["neutral", "positive", "negative"]


class SentimentOutput(BaseModel):
    decisions: dict[str, SentimentDecision] = Field(description="Dictionary of ticker news' sentiment")


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    assets = data.get("assets", "A")
    meta = state.get("metadata", {})
    model_name = meta.get("model_name", "qwen-max-latest")
    model_provider = meta.get("model_provider", "QWen")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, assets, end_date, limit=100)

        # Create the prompt template
        template = ChatPromptTemplate.from_messages(
           [
            (
              "system",
              """You are a stock and futures analyst who can analyze news about a specified stock or futures
              and determine the sentiment as neutral, positive, or negative.
              """,
            ),
            (
              "human",
              """Based on the news, make your decisions.

              Here are the news about ticker:
              {ticker_news}

              Output strictly in JSON with the following structure:
              {{
                "decisions": {{
                    "news1": {{
                        "sentiment": "neutral/positive/negative"
                    }},
                    "news2": {{
                        ...
                    }}
                }}
              }}
              """,
            ),
            ]
        )

        # Generate the prompt
        prompt = template.invoke(
            {
                "ticker_news": company_news,
            }
        )

        # Create default factory for SentimentOutput
        def create_default_sentiment_output():
            return SentimentOutput(decisions={news: SentimentDecision(sentiment="neutral") for news in company_news})

        ret = call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=SentimentOutput, agent_name="sentiment_agent", default_factory=create_default_sentiment_output)
        sentiment = pd.Series([n.sentiment for n in ret.decisions.values()]).dropna()

        news_signals = np.where(sentiment == "negative", "bearish",
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()

        progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7

        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }
