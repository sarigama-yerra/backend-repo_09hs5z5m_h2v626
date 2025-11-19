import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# Database helpers
from database import db, create_document, get_documents

app = FastAPI(title="ATLAS NΞO Local Services", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------
# Models (request/response)
# ---------------------------------------------
class StrategyPayload(BaseModel):
    name: str
    description: Optional[str] = None
    mode: str = Field("paper", description="paper | live")
    blocks: Dict[str, Any] = Field(default_factory=dict)
    indicators: List[Dict[str, Any]] = Field(default_factory=list)
    ml: List[Dict[str, Any]] = Field(default_factory=list)


class OrderInternal(BaseModel):
    side: str  # buy | sell
    symbol: str
    size: float
    type: str  # market | limit | stop | stop-limit
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    leverage: Optional[float] = None
    broker: str = Field("paper", description="target broker adapter id (e.g., alpaca, binance, oanda, ibkr, ccxt, kraken, paper)")


class RiskConstraints(BaseModel):
    max_daily_loss: float = 100.0
    max_position_size: float = 1.0
    allowed_symbols: List[str] = Field(default_factory=list)
    leverage_limit: float = 5.0
    max_exposure: float = 2.0


class SignalRequest(BaseModel):
    symbol: str
    features: Dict[str, float] = Field(default_factory=dict)
    mode: str = Field("neutral", description="conservative | neutral | aggressive | adaptive | shadow")


# ---------------------------------------------
# Health & Diagnostics
# ---------------------------------------------
@app.get("/")
def read_root():
    return {"service": "ATLAS NΞO Local", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", "unknown")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# ---------------------------------------------
# Unified Asset Universe (mocked starter set)
# ---------------------------------------------
@app.get("/api/symbols")
def get_symbols() -> List[Dict[str, Any]]:
    # Minimal seed universe; in production this comes from Data Feed service
    symbols = [
        {"symbol": "EURUSD", "asset": "forex", "broker_availability": ["oanda", "ibkr"], "leverage": 30, "session": "24/5"},
        {"symbol": "BTCUSDT", "asset": "crypto", "broker_availability": ["binance", "kraken", "ccxt"], "leverage": 5, "session": "24/7"},
        {"symbol": "AAPL", "asset": "stock", "broker_availability": ["alpaca", "ibkr"], "leverage": 2, "session": "US"},
        {"symbol": "US500", "asset": "index", "broker_availability": ["ibkr"], "leverage": 20, "session": "US"},
    ]
    # Normalize minimal fields per spec
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    normalized = []
    for s in symbols:
        normalized.append({
            "symbol": s["symbol"],
            "timestamp": now,
            "price": None,
            "bid": None,
            "ask": None,
            "volume": None,
            "spread": None,
            "meta": s,
        })
    return normalized


# ---------------------------------------------
# Strategy Sandbox: Save/Load Strategies
# ---------------------------------------------
@app.post("/api/strategies")
def save_strategy(payload: StrategyPayload):
    doc = payload.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    strategy_id = create_document("strategy", doc)
    return {"id": strategy_id, "status": "saved"}


@app.get("/api/strategies")
def list_strategies(limit: int = 50):
    items = get_documents("strategy", {}, limit)
    # Convert ObjectIds to strings for client
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items


# ---------------------------------------------
# AI Core Engine (mocked blending)
# ---------------------------------------------
@app.post("/api/ai/signal")
def ai_signal(req: SignalRequest):
    # Very lightweight heuristic placeholder (deterministic for demo)
    base = sum((req.features.get(k, 0.0) for k in sorted(req.features.keys())))
    confidence = min(0.99, max(0.01, abs(base) % 1.0))
    if req.mode == "conservative":
        confidence *= 0.6
    elif req.mode == "aggressive":
        confidence = min(0.99, confidence * 1.25)
    direction = "buy" if ((base * 1000) % 2) > 1 else "sell"
    return {
        "symbol": req.symbol,
        "signal": direction,
        "confidence": round(confidence, 3),
        "blending": {
            "lstm": round(confidence * 0.4, 3),
            "regime": round(confidence * 0.25, 3),
            "structure": round(confidence * 0.2, 3),
            "pattern": round(confidence * 0.15, 3),
            "method": "confidence-fusion"
        }
    }


# ---------------------------------------------
# Risk Composer
# ---------------------------------------------
@app.post("/api/risk/validate")
def risk_validate(order: OrderInternal, rules: RiskConstraints = RiskConstraints()):
    # Check symbol allow-list
    if rules.allowed_symbols and order.symbol not in rules.allowed_symbols:
        return {"approved": False, "action": "block", "reason": "symbol_not_allowed"}

    # Position size
    if order.size > rules.max_position_size:
        # Might reduce size in production; here return suggestion
        return {"approved": False, "action": "reduce_size", "max_size": rules.max_position_size, "reason": "position_size_exceeds_limit"}

    # Leverage
    if order.leverage and order.leverage > rules.leverage_limit:
        return {"approved": False, "action": "cap_leverage", "max_leverage": rules.leverage_limit, "reason": "leverage_exceeds_limit"}

    return {"approved": True, "action": "pass"}


# ---------------------------------------------
# Trade Execution Layer (Broker Gateway transform stub)
# ---------------------------------------------

def transform_to_broker(order: OrderInternal) -> Dict[str, Any]:
    # Minimal mapping examples for demo purposes
    base = {
        "side": order.side,
        "symbol": order.symbol,
        "qty": order.size,
        "type": order.type,
        "price": order.price,
        "sl": order.sl,
        "tp": order.tp,
        "leverage": order.leverage,
    }
    broker = order.broker.lower()
    if broker == "alpaca":
        return {"symbol": order.symbol, "qty": order.size, "side": order.side, "type": "market" if order.type == "market" else "limit", "time_in_force": "gtc", "limit_price": order.price}
    if broker == "binance":
        return {"symbol": order.symbol, "side": order.side.upper(), "type": order.type.replace("-", "_").upper(), "quantity": order.size, "price": order.price}
    if broker == "oanda":
        return {"instrument": order.symbol, "units": int(order.size if order.side == "buy" else -order.size), "type": order.type.upper(), "price": order.price}
    if broker == "ibkr":
        return {"conid": order.symbol, "action": order.side.upper(), "orderType": order.type.upper(), "totalQuantity": order.size, "lmtPrice": order.price}
    if broker == "kraken":
        return {"pair": order.symbol, "type": order.side, "ordertype": order.type, "volume": order.size, "price": order.price}
    if broker == "ccxt":
        return {"symbol": order.symbol, "type": order.type, "side": order.side, "amount": order.size, "price": order.price}
    # default/paper
    return {**base, "route": "paper"}


@app.post("/api/order/route")
def route_order(order: OrderInternal):
    # 1) Risk check (basic pass-through for demo)
    risk = risk_validate(order)
    if not risk.get("approved", False):
        return {"status": "blocked", "risk": risk}

    # 2) Transform to broker format (atlas-broker-gateway responsibility in full app)
    payload = transform_to_broker(order)

    # 3) Write to structured log
    log_entry = {
        "type": "order_routed",
        "broker": order.broker,
        "internal": order.model_dump(),
        "transformed": payload,
        "timestamp": datetime.now(timezone.utc),
    }
    create_document("log", log_entry)

    # 4) Simulated broker response
    return {"status": "accepted", "broker": order.broker, "payload": payload, "id": create_document("order", order.model_dump())}


# ---------------------------------------------
# Logger Service (basic list endpoint)
# ---------------------------------------------
@app.get("/api/logs")
def list_logs(limit: int = 50):
    items = get_documents("log", {}, limit)
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
