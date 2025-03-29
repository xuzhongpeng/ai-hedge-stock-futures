from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import json

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 用于表单安全验证

# API URL
API_URL = "http://localhost:8086/api/analysis"

# 分析师列表
ANALYSTS = [
    "ben_graham",
    "bill_ackman",
    "cathie_wood",
    "charlie_munger",
    "phil_fisher",
    "stanley_druckenmiller",
    "warren_buffett",
    "technical_analyst",
    "fundamentals_analyst",
    "sentiment_analyst",
    "valuation_analyst"
]

def format_analysis_results(result, ticker):
    if not result or "analyst_signals" not in result:
        return "No analysis results returned"

    html_output = f"<h1>Analysis Results for {ticker.upper()}</h1>"

    # 决策部分
    if "decisions" in result and ticker in result["decisions"]:
        decision = result["decisions"][ticker]
        html_output += f"""
        <div class="decision-box">
            <h2>Final Decision</h2>
            <p><strong>Action:</strong> {decision.get('action', 'N/A').upper()}</p>
            <p><strong>Confidence:</strong> {decision.get('confidence', 'N/A')}%</p>
            <p><strong>Quantity:</strong> {decision.get('quantity', 'N/A')}</p>
            <p><strong>Reasoning:</strong> {decision.get('reasoning', 'N/A')}</p>
        </div>
        """

    # 分析师信号
    html_output += "<h2>Analyst Signals</h2>"

    analyst_signals = result.get("analyst_signals", {})
    for analyst, data in analyst_signals.items():
        if ticker not in data:
            continue
        signal_data = data[ticker]

        # 特殊处理风险控制
        if analyst == "risk_management_agent":
            html_output += f"""
            <div class="analyst-box">
                <h3>{format_analyst_name(analyst)}</h3>
                <p><strong>Current Price:</strong> ${signal_data.get('current_price', 'N/A')}</p>
                <p><strong>Remaining Position Limit:</strong> ${signal_data.get('remaining_position_limit', 'N/A')}</p>
                <div class="reasoning">
                    <ul>
                        <li>Available Cash: ${signal_data.get('reasoning', {}).get('available_cash', 'N/A')}</li>
                        <li>Portfolio Value: ${signal_data.get('reasoning', {}).get('portfolio_value', 'N/A')}</li>
                    </ul>
                </div>
            </div>
            """
            continue

        # 格式化其他分析师
        signal = signal_data.get('signal', 'N/A')
        confidence = signal_data.get('confidence', 'N/A')
        reasoning = signal_data.get('reasoning', 'N/A')

        html_output += f"""
        <div class="analyst-box {signal}">
            <h3>{format_analyst_name(analyst)}</h3>
            <p><strong>Signal:</strong> {signal.upper()}</p>
            <p><strong>Confidence:</strong> {confidence}%</p>
            <div class="reasoning">
                <p><strong>Reasoning:</strong> {reasoning}</p>
            </div>
        </div>
        """

    return html_output

def format_analyst_name(analyst_key):
    """将分析师名称格式化"""
    name = analyst_key.replace("_agent", "").replace("_analyst", "")
    return " ".join(word.capitalize() for word in name.split("_"))

@app.route("/", methods=["GET", "POST"])
def index():
    """主页路由"""
    if request.method == "POST":
        ticker = request.form.get("ticker").strip().upper()
        assets = request.form.get("selected_filter").strip().split(" ")[0]
        selected_analysts = request.form.getlist("selected_analysts")

        if not ticker:
            flash("Please enter a stock ticker", "error")
            return redirect(url_for("index"))

        if not selected_analysts:
            flash("Please select at least one analyst", "error")
            return redirect(url_for("index"))

        # 调用 API 分析
        payload = {
            "tickers": ticker,
            "assets": assets,
            "selectedAnalysts": selected_analysts,
            "modelName": "qwen-max-latest"
        }

        print(payload)
        try:
            response = requests.post(API_URL, json=payload, timeout=10000)
            response.raise_for_status()
            result = response.json()

            # 格式化结果
            formatted_result = format_analysis_results(result, ticker)
            return render_template("index.html",
                                   results=formatted_result,
                                   ticker=ticker,
                                   selected_analysts=selected_analysts,
                                   ANALYSTS=ANALYSTS)

        except requests.exceptions.RequestException as e:
            flash(f"API Error: {str(e)}", "error")
        except Exception as e:
            flash(f"Unexpected Error: {str(e)}", "error")

        return redirect(url_for("index"))

    return render_template("index.html", ANALYSTS=ANALYSTS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
