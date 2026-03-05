"""API routes for real-time trading agent."""

from typing import Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
import json

from backend.app.schemas.trading import (
    AgentConfigRequest,
    AgentStatus,
    AgentStartResponse,
    AgentStopResponse,
    TradeHistoryResponse,
    OrderHistoryResponse,
    AlertHistoryResponse,
    PortfolioHistoryResponse,
    RiskMetricsResponse,
    ManualOrderRequest,
    ClosePositionRequest,
    Trade,
    PerformanceMetrics,
)
from backend.app.services.trading_service import get_trading_service

router = APIRouter()


# ============== Agent Management ==============

@router.get("/agent/status", response_model=AgentStatus)
async def get_agent_status():
    """
    Get the current trading agent status.

    Returns comprehensive status including:
    - Running state and configuration
    - Portfolio value and positions
    - Recent signals and trades
    - Performance metrics
    - Market data
    """
    service = get_trading_service()
    status = service.get_status()

    if status is None:
        return AgentStatus(is_running=False, trading_mode='paper', ticker='^GSPC')

    return status


@router.post("/agent/start", response_model=AgentStartResponse)
async def start_agent(config: AgentConfigRequest):
    """
    Start the trading agent with the specified configuration.

    The agent will:
    - Load the specified ML model
    - Begin fetching market data
    - Generate trading signals
    - Execute trades (paper trading mode)
    - Track performance metrics

    Note: Only one agent can run at a time. Starting a new agent
    will stop any existing agent.
    """
    service = get_trading_service()

    try:
        agent_id, agent = service.start_agent(config)

        return AgentStartResponse(
            success=True,
            message=f"Trading agent started in {config.trading_mode.value} mode",
            agent_id=agent_id,
            config={
                'model_key': config.model_key,
                'ticker': config.ticker,
                'trading_mode': config.trading_mode.value,
                'initial_capital': config.initial_capital,
                'signal_threshold': config.signal_threshold,
                'max_position_size': config.max_position_size,
                'min_confidence': config.min_confidence,
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct,
                'position_sizing': config.position_sizing.value,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/stop", response_model=AgentStopResponse)
async def stop_agent():
    """
    Stop the currently running trading agent.

    Returns the final portfolio state including:
    - Final portfolio value
    - Total P&L
    - Total trades executed
    """
    service = get_trading_service()

    result = service.stop_agent()

    if result is None:
        return AgentStopResponse(
            success=False,
            message="No agent is currently running",
            final_portfolio_value=0.0,
            total_pnl=0.0,
            total_trades=0
        )

    return AgentStopResponse(**result)


@router.get("/agent/running")
async def is_agent_running():
    """Check if the trading agent is currently running."""
    service = get_trading_service()
    return {"is_running": service.is_running()}


# ============== Trade History ==============

@router.get("/trades", response_model=TradeHistoryResponse)
async def get_trade_history(
    page: int = 1,
    page_size: int = 50
):
    """
    Get trade history for the current agent.

    Returns paginated list of executed trades with:
    - Trade details (ticker, side, quantity, price)
    - P&L information
    - Model and signal confidence
    """
    service = get_trading_service()
    result = service.get_trades(page, page_size)

    # Calculate summary metrics
    trades = result.get('trades', [])
    total_pnl = sum(t.pnl for t in trades) if trades else 0.0
    winning = len([t for t in trades if t.pnl > 0])
    losing = len([t for t in trades if t.pnl < 0])
    win_rate = winning / len(trades) * 100 if trades else 0.0

    summary = PerformanceMetrics(
        total_trades=len(trades),
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        total_pnl=total_pnl,
    )

    return TradeHistoryResponse(
        trades=result.get('trades', []),
        total=result.get('total', 0),
        page=page,
        page_size=page_size,
        summary=summary
    )


@router.get("/orders", response_model=OrderHistoryResponse)
async def get_order_history(
    page: int = 1,
    page_size: int = 50
):
    """Get order history for the current agent."""
    service = get_trading_service()
    result = service.get_orders(page, page_size)

    return OrderHistoryResponse(
        orders=result.get('orders', []),
        total=result.get('total', 0),
        page=page,
        page_size=page_size
    )


# ============== Alerts ==============

@router.get("/alerts", response_model=AlertHistoryResponse)
async def get_alerts(limit: int = 50):
    """
    Get recent alerts from the trading agent.

    Alerts include:
    - Signal generation events
    - Risk management triggers (stop-loss, take-profit)
    - Execution notifications
    - Error messages
    """
    service = get_trading_service()
    result = service.get_alerts(limit)

    return AlertHistoryResponse(
        alerts=result.get('alerts', []),
        total=result.get('total', 0),
        unread_count=result.get('unread_count', 0)
    )


# ============== Manual Orders ==============

@router.post("/orders/manual")
async def place_manual_order(request: ManualOrderRequest):
    """
    Place a manual order.

    Allows manual trading alongside automated signals.
    Order will be executed at current market price.
    """
    service = get_trading_service()

    if not service.is_running():
        raise HTTPException(status_code=400, detail="Agent is not running")

    result = service.place_manual_order(
        ticker=request.ticker,
        side=request.side.value,
        quantity=request.quantity
    )

    if result is None:
        raise HTTPException(status_code=400, detail="Order could not be executed")

    return {"success": True, "trade": result}


@router.post("/positions/close")
async def close_position(request: ClosePositionRequest):
    """
    Close an existing position.

    Will sell all shares if quantity not specified.
    """
    service = get_trading_service()

    if not service.is_running():
        raise HTTPException(status_code=400, detail="Agent is not running")

    result = service.close_position(
        ticker=request.ticker,
        quantity=request.quantity
    )

    if result is None:
        raise HTTPException(status_code=400, detail="No position to close")

    return {"success": True, "trade": result}


# ============== Portfolio ==============

@router.get("/portfolio/history", response_model=PortfolioHistoryResponse)
async def get_portfolio_history():
    """
    Get portfolio value history over time.

    Returns time series of:
    - Total portfolio value
    - Cash and positions breakdown
    - Daily and cumulative returns
    """
    service = get_trading_service()
    status = service.get_status()

    if status is None or not status.is_running:
        return PortfolioHistoryResponse(
            history=[],
            start_date="",
            end_date="",
            initial_capital=100000.0,
            final_value=100000.0,
            total_return=0.0
        )

    agent = service._active_agent
    if agent is None:
        return PortfolioHistoryResponse(
            history=[],
            start_date="",
            end_date="",
            initial_capital=100000.0,
            final_value=100000.0,
            total_return=0.0
        )

    history = agent.portfolio_history

    if not history:
        return PortfolioHistoryResponse(
            history=[],
            start_date="",
            end_date="",
            initial_capital=agent.initial_capital,
            final_value=agent.get_portfolio_value(),
            total_return=0.0
        )

    return PortfolioHistoryResponse(
        history=history,
        start_date=history[0]['timestamp'] if history else "",
        end_date=history[-1]['timestamp'] if history else "",
        initial_capital=agent.initial_capital,
        final_value=agent.get_portfolio_value(),
        total_return=((agent.get_portfolio_value() / agent.initial_capital) - 1) * 100
    )


@router.get("/portfolio/positions")
async def get_positions():
    """Get current portfolio positions."""
    service = get_trading_service()
    status = service.get_status()

    if status is None:
        return {"positions": [], "total_value": 0.0}

    return {
        "positions": status.positions,
        "total_value": status.positions_value,
        "cash": status.cash
    }


# ============== Risk Metrics ==============

@router.get("/risk/metrics")
async def get_risk_metrics():
    """
    Calculate current risk metrics for the portfolio.

    Includes:
    - Value at Risk (VaR) at 95% and 99% confidence
    - Maximum drawdown
    - Volatility
    - Sharpe ratio
    """
    service = get_trading_service()
    status = service.get_status()

    if status is None or status.performance is None:
        return {
            "var_95": 0.0,
            "var_99": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "current_drawdown": 0.0
        }

    return {
        "var_95": status.total_value * 0.02,  # Simplified VaR estimate
        "var_99": status.total_value * 0.03,
        "max_drawdown": status.performance.max_drawdown,
        "volatility": 0.0,  # Would need historical returns
        "sharpe_ratio": status.performance.sharpe_ratio,
        "current_drawdown": status.performance.max_drawdown
    }


# ============== Price History ==============

@router.get("/price/history")
async def get_price_history():
    """
    Get real-time market price history collected by the trading agent.

    Returns time series of OHLCV price data for the active ticker,
    used to render the market price chart on the trading page.
    """
    service = get_trading_service()
    agent = service._active_agent

    if agent is None:
        return {"prices": [], "ticker": "^GSPC"}

    return {
        "prices": list(agent.price_history),
        "ticker": agent.config.ticker,
    }


# ============== WebSocket for Real-time Updates ==============

@router.websocket("/ws/updates")
async def websocket_trading_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading updates.

    Streams:
    - Agent status every 2 seconds
    - Immediate notifications for trades and alerts
    """
    await websocket.accept()

    service = get_trading_service()

    try:
        while True:
            # Get current status
            status = service.get_status()

            if status:
                # Send status update
                await websocket.send_json({
                    "type": "status",
                    "data": status.model_dump()
                })

            # Wait before next update
            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass


@router.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal notifications.

    Streams trading signals as they are generated.
    """
    await websocket.accept()

    service = get_trading_service()
    last_signal_count = 0

    try:
        while True:
            status = service.get_status()

            if status and status.is_running:
                current_count = status.total_signals

                if current_count > last_signal_count and status.signals:
                    # New signal(s) generated
                    new_signals = status.signals[:current_count - last_signal_count]
                    for signal in new_signals:
                        await websocket.send_json({
                            "type": "signal",
                            "data": signal.model_dump()
                        })

                last_signal_count = current_count

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
