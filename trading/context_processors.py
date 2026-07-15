from trading.models import StrategyConfig


def strategy_config(request):
    try:
        config = StrategyConfig.get_active()
    except Exception:
        config = None
    return {"active_config": config}