import argparse
from orchestrator import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Stock Forecast Pipeline Trigger")
    parser.add_argument("--ticker", type=str, help="Ticker to process")
    args = parser.parse_args()
    
    run_pipeline(args.ticker)

if __name__ == "__main__":
    main()
