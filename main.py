import os
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Database helpers (MongoDB via pymongo). Environment: DATABASE_URL, DATABASE_NAME
from database import db, create_document, get_documents

app = FastAPI(title="ATLAS NΞO Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class IndicatorConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class StrategyBlock(BaseModel):
    id: str
    type: str
    config: Dict[str, Any] = {}

class StrategyPayload(BaseModel):
    name: str
    description: Optional[str] = None
    mode: Literal["paper", "live", "backtest"] = "paper"
    blocks: List[StrategyBlock] = []
    indicators: List[IndicatorConfig] = []
    ml: Dict[str, Any] = {}

class OrderInternal(BaseModel):
    side: Literal["buy", "sell"]
    symbol: str
    size: float
    type: Literal["market", "limit"] = "market"
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    leverage: Optional[float] = 1.0
    broker: str = "paper"

class RiskConstraints(BaseModel):
    max_size: Optional[float] = None
    max_leverage: Optional[float] = None
    allow_symbols: Optional[List[str]] = None

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1m"
    features: Dict[str, Any] = {}


# -----------------------------
# Utilities
# -----------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def transform_to_broker(order: OrderInternal) -> Dict[str, Any]:
    # Stub mapping for multiple brokers
    payload = order.model_dump()
    broker = order.broker.lower()
    if broker == "alpaca":
        return {
            "symbol": order.symbol,
            "qty": order.size,
            "side": order.side,
            "type": order.type,
            "time_in_force": "gtc",
            **({"limit_price": order.price} if order.type == "limit" and order.price else {}),
        }
    if broker == "binance":
        return {
            "symbol": order.symbol.replace("/", ""),
            "side": order.side.upper(),
            "type": "MARKET" if order.type == "market" else "LIMIT",
            "quantity": order.size,
            **({"price": order.price} if order.type == "limit" and order.price else {}),
        }
    if broker == "oanda":
        return {
            "instrument": order.symbol,
            "units": int(order.size if order.side == "buy" else -order.size),
            "type": order.type,
            **({"price": order.price} if order.type == "limit" and order.price else {}),
        }
    if broker in {"ibkr", "kraken", "ccxt"}:
        return {"unified": payload}
    # default paper broker payload
    return {"paper": payload}


# -----------------------------
# Core endpoints
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "ATLAS NΞO Backend is running", "time": now_iso()}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database_url"] = "✅ Set"
            response["database_name"] = getattr(db, "name", "unknown")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    # env check
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else response["database_url"]
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else response["database_name"]
    return response


@app.get("/api/symbols")
def get_symbols():
    # Mocked, normalized universe
    return {
        "symbols": [
            {"symbol": "EUR/USD", "asset": "forex", "tick": 0.0001, "min_size": 1000},
            {"symbol": "BTC/USDT", "asset": "crypto", "tick": 0.01, "min_size": 0.001},
            {"symbol": "AAPL", "asset": "stock", "tick": 0.01, "min_size": 1},
            {"symbol": "ES", "asset": "futures", "tick": 0.25, "min_size": 1},
            {"symbol": "SPY", "asset": "etf", "tick": 0.01, "min_size": 1},
        ]
    }


# Strategies
@app.post("/api/strategies")
def save_strategy(payload: StrategyPayload):
    data = payload.model_dump()
    data["created_at"] = now_iso()
    data["updated_at"] = now_iso()
    try:
        inserted_id = create_document("strategy", data)
        return {"ok": True, "id": inserted_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
def list_strategies():
    try:
        docs = get_documents("strategy", limit=50)
        # Convert ObjectId to string
        sanitized = []
        for d in docs:
            d["_id"] = str(d.get("_id"))
            sanitized.append(d)
        return {"items": sanitized}
    except Exception as e:
        # if db not configured, return empty list gracefully
        return {"items": []}


# AI signal (mock)
@app.post("/api/ai/signal")
def ai_signal(req: SignalRequest):
    # Simple mock blending
    import math
    base = hash(req.symbol + req.timeframe) % 100 / 100.0
    signal = (math.sin(base * 3.14) + 1) / 2
    confidence = 0.5 + (base / 2)
    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "signal": round(signal, 3),
        "confidence": round(confidence, 3),
        "components": {
            "structure": round(signal * 0.4, 3),
            "pattern": round(signal * 0.3, 3),
            "regime": round(signal * 0.3, 3),
        },
        "time": now_iso(),
    }


# Risk validation
@app.post("/api/risk/validate")
def risk_validate(order: OrderInternal, rules: RiskConstraints = RiskConstraints()):
    errors = []
    if rules.max_size is not None and order.size > rules.max_size:
        errors.append(f"size {order.size} exceeds max_size {rules.max_size}")
    if rules.max_leverage is not None and (order.leverage or 1) > rules.max_leverage:
        errors.append(f"leverage {order.leverage} exceeds max_leverage {rules.max_leverage}")
    if rules.allow_symbols is not None and order.symbol not in rules.allow_symbols:
        errors.append(f"symbol {order.symbol} not allowed")
    return {"ok": len(errors) == 0, "errors": errors}


# Order routing (stub)
@app.post("/api/order/route")
def order_route(order: OrderInternal):
    routed_payload = transform_to_broker(order)
    log_entry = {
        "type": "order",
        "broker": order.broker,
        "order": order.model_dump(),
        "payload": routed_payload,
        "status": "accepted",
        "time": now_iso(),
    }
    try:
        create_document("logs", log_entry)
    except Exception:
        pass
    return {"ok": True, "status": "accepted", "routed": routed_payload}


@app.get("/api/logs")
def get_logs(limit: int = 50):
    try:
        docs = get_documents("logs", limit=limit)
        for d in docs:
            d["_id"] = str(d.get("_id"))
        return {"items": docs}
    except Exception:
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
