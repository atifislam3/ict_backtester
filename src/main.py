import os
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.events import EventBus, CandleEvent, PatternEvent
from src.pattern_detector import PatternDetector
from src.backtest_engine import EventDrivenBacktester
from src.statistics import calculate_statistics
from src.visualizer import generate_html_report
from src.models import Candle

app = typer.Typer(help="ICT Backtester Pro CLI")
console = Console()

@app.command()
def run(
    config: str = typer.Option("config/default.yaml", help="Path to config YAML file"),
    symbol: Optional[str] = typer.Option(None, help="Override instrument symbol"),
    start: Optional[str] = typer.Option(None, help="Override start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="Override end date (YYYY-MM-DD)")
):
    console.print("[bold cyan]Initializing ICT Backtester Pro...[/bold cyan]")
    
    # 1. Load config
    cm = ConfigManager(config)
    cfg = cm.load_config()
    
    # Apply CLI overrides
    if start or end:
        cfg = cfg.model_copy(update={
            'date_range': cfg.date_range.model_copy(update={'start': start or cfg.date_range.start, 'end': end or cfg.date_range.end})
        })
    if symbol:
        cfg = cfg.model_copy(update={
            'instrument': cfg.instrument.model_copy(update={'symbol': symbol, 'display_name': symbol.split('=')[0]})
        })
        
    # 2. Download Data
    dl = DataLoader()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Downloading {cfg.instrument.symbol} data...", total=None)
        df = dl.get_data(
            instrument=cfg.instrument.symbol, 
            timeframe=cfg.timeframe, 
            start_date=cfg.date_range.start, 
            end_date=cfg.date_range.end
        )
        
    console.print(f"[green]✓ Data loaded: {len(df)} candles.[/green]")
    
    # Convert to candles
    candles = []
    for idx, row in df.iterrows():
        candles.append(Candle(
            timestamp=idx,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row.get('volume', 0.0)
        ))
        
    # 3. Detect Patterns & Run Backtest
    bus = EventBus()
    detector = PatternDetector(bus, cfg.patterns)
    engine = EventDrivenBacktester(bus, cfg)
    
    all_patterns = []
    bus.subscribe(PatternEvent, lambda e: all_patterns.append(e))
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Running Pattern Scan & Backtest Walk...", total=len(candles))
        
        for c in candles:
            bus.emit(CandleEvent(candle=c))
            progress.advance(task)
            
    trades, equity_df, order_df = engine.get_results()
    console.print(f"[green]✓ Backtest completed. {len(trades)} trades executed.[/green]")
    
    # 4. Compute Stats
    console.print("[cyan]Computing statistics...[/cyan]")
    stats = calculate_statistics(trades, initial_capital=10000.0)
    
    # 5. Generate Report
    console.print("[cyan]Generating HTML Report...[/cyan]")
    report_path = generate_html_report(
        candles=candles,
        swings=detector.swings,
        patterns=all_patterns,
        trades=trades,
        equity_df=equity_df,
        stats=stats,
        output_dir=cfg.reporting.output_dir
    )
    console.print(f"[green]✓ Interactive HTML Report generated: {report_path}[/green]")
    
    if cfg.reporting.save_json:
        os.makedirs(cfg.reporting.output_dir, exist_ok=True)
        json_path = os.path.join(cfg.reporting.output_dir, "report.json")
        with open(json_path, "w") as f:
            f.write(stats.model_dump_json(indent=2))
        console.print(f"[green]✓ Saved JSON report to {json_path}[/green]")
        
    # 6. Terminal Summary
    table = Table(title="Backtest Performance Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Trades", str(stats.total_trades))
    table.add_row("Win Rate", f"{stats.win_rate*100:.2f}%")
    pf = "Infinity" if stats.profit_factor == float('inf') else f"{stats.profit_factor:.2f}"
    table.add_row("Profit Factor", pf)
    table.add_row("Max Drawdown", f"{stats.max_drawdown_pct:.2f}%")
    table.add_row("Expectancy", f"${stats.expectancy:.2f}")
    table.add_row("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
    
    console.print(table)

if __name__ == "__main__":
    app()
