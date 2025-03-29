import os
import sys
import math
import json
import threading
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from datetime import datetime
from dateutil.relativedelta import relativedelta
from main import run_hedge_fund

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)

websocket_clients = []

@app.route('/api/analysis', methods=['POST'])
def run_analysis():
    """執行對股票的分析"""
    try:
        data = request.get_json()
        ticker_list = data.get('tickers', 'AAPL').split(',')
        assets=data.get('assets', "US")
        selected_analysts = data.get('selectedAnalysts', [])
        model_name = data.get('modelName')

        end_date = data.get('endDate') or datetime.now().strftime('%Y-%m-%d')
        start_date = data.get('startDate') or (datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(months=3)).strftime('%Y-%m-%d')

        portfolio = {
            "cash": data.get('initialCash', 100000),
            "positions": {},
            "cost_basis": {},
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in ticker_list}
        }

        result = run_hedge_fund(
            tickers=ticker_list,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=True,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider="QWen"
        )

        def convert_nan_to_null(obj):
            if isinstance(obj, dict):
                return {k: convert_nan_to_null(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan_to_null(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj

        result = convert_nan_to_null(result)
        return json.dumps(result)
    except Exception as e:
        error_message = f"API Error in run_analysis: {str(e)}"
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@sock.route('/ws/logs')
def logs(ws):
    websocket_clients.append(ws)
    try:
        while True:
            ws.receive()
    except Exception:
        websocket_clients.remove(ws)

if __name__ == "__main__":
    api_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8086, "debug": True, "use_reloader": False})
    api_thread.daemon = True
    api_thread.start()
    print("API Server started on http://localhost:8086")
    api_thread.join()
