import os
import sys
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

def broadcast_log(message, level="info"):
    log_data = {"level": level, "message": message}
    for client in websocket_clients[:]:
        try:
            client.send(json.dumps(log_data))
        except Exception:
            websocket_clients.remove(client)

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
        print(ticker_list)
        print(assets)

        broadcast_log(f"Starting analysis for {ticker_list}", "info")
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

        broadcast_log("Analysis completed successfully", "success")
        return jsonify(result)

    except Exception as e:
        error_message = f"API Error: {str(e)}"
        broadcast_log(error_message, "error")
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
