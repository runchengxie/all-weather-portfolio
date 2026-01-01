# 全天候投资组合回测

这个项目提供一个简洁的「All Weather/全天候」资产配置回测脚本，支持风险平价、逆波动与固定权重，并可进行目标波动率控制与杠杆上限约束。

## 功能
- 组合权重：`risk-parity`、`inverse-vol`、`fixed`
- 波动率目标：按滚动窗口估计组合波动，控制目标波动与最大杠杆
- 数据来源：Stooq（默认，无需 API key）或 Alpaca（需要 API key）
- 可选输出：权重序列、组合收益、净值曲线与运行元信息

## 环境要求
- Python 3.9+
- 依赖：`numpy`、`pandas`、`requests`

## 安装
```bash
pip install -e .
```

## 快速开始
```bash
# 默认参数（Stooq，风险平价）
python all_weather_backtest.py

# 指定区间与输出目录
python all_weather_backtest.py --start 2005-01-01 --end 2024-12-31 --out out/

# 固定权重组合
python all_weather_backtest.py \
  --portfolio fixed \
  --fixed-weights "SPY=0.3,TLT=0.4,IEF=0.15,GLD=0.075,DBC=0.075"
```

## 使用 Alpaca 数据
```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."

python all_weather_backtest.py --data-source alpaca --alpaca-feed iex
```

## 关键参数说明
- `--data-source`：`stooq` 或 `alpaca`
- `--tickers`：逗号分隔 ETF 列表，例如 `SPY,TLT,IEF,GLD,DBC`
- `--portfolio`：`risk-parity`、`inverse-vol`、`fixed`
- `--fixed-weights`：固定权重字符串（会自动归一化）
- `--lookback`：回看窗口（交易日）
- `--rebalance`：再平衡频率（pandas offset alias，如 `M`、`W-FRI`）
- `--target-vol`：目标年化波动率
- `--max-leverage`：最大总杠杆
- `--out`：输出目录

## 输出文件（当使用 `--out`）
- `portfolio_returns.csv`：组合日收益
- `weights_daily.csv`：日度权重
- `equity_curve.csv`：净值曲线
- `run_meta.json`：本次运行参数信息

## 测试
```bash
pip install -e ".[dev]"
./scripts/test.sh
```

## 备注
- 这是对 Bridgewater All Weather 思路的 ETF 代理回测。
- 未考虑交易成本、税费、滑点与融资成本。
- 数据质量依赖于数据源，仅供学习研究，不构成投资建议。
