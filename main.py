"""
Currency & Crypto Tracker MCP Agent
Provides real-time currency conversion, stock data, and crypto prices for trading decisions
"""

import asyncio
import json
from typing import Any, Optional
from datetime import datetime
import httpx
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import AnyUrl
import mcp.server.stdio

# Initialize MCP server
app = Server("currency-crypto-tracker")

# API Configuration (use free tier APIs)
EXCHANGE_RATE_API = "https://api.exchangerate-api.com/v4/latest"
CRYPTO_API = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_KEY = "CCXNJH50RGJZA53U"  # Replace with your free key from https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API = "https://www.alphavantage.co/query"


class TradingDataFetcher:
    """Handles API calls to various financial data sources"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> dict:
        """Get exchange rate between two currencies"""
        try:
            url = f"{EXCHANGE_RATE_API}/{from_currency.upper()}"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            rate = data["rates"].get(to_currency.upper())
            if not rate:
                return {"error": f"Currency {to_currency} not found"}
            
            return {
                "from": from_currency.upper(),
                "to": to_currency.upper(),
                "rate": rate,
                "date": data.get("date"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> dict:
        """Convert amount from one currency to another"""
        rate_data = await self.get_exchange_rate(from_currency, to_currency)
        
        if "error" in rate_data:
            return rate_data
        
        converted = amount * rate_data["rate"]
        return {
            "original_amount": amount,
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "converted_amount": round(converted, 2),
            "exchange_rate": rate_data["rate"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_crypto_price(self, crypto_id: str, vs_currency: str = "usd") -> dict:
        """Get current crypto price and market data"""
        try:
            url = f"{CRYPTO_API}/simple/price"
            params = {
                "ids": crypto_id.lower(),
                "vs_currencies": vs_currency.lower(),
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true"
            }
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if crypto_id.lower() not in data:
                return {"error": f"Crypto {crypto_id} not found"}
            
            crypto_data = data[crypto_id.lower()]
            return {
                "crypto": crypto_id,
                "currency": vs_currency.upper(),
                "price": crypto_data.get(vs_currency.lower()),
                "24h_change": crypto_data.get(f"{vs_currency.lower()}_24h_change"),
                "market_cap": crypto_data.get(f"{vs_currency.lower()}_market_cap"),
                "24h_volume": crypto_data.get(f"{vs_currency.lower()}_24h_vol"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_crypto_market_data(self, crypto_id: str) -> dict:
        """Get detailed market data for a cryptocurrency"""
        try:
            url = f"{CRYPTO_API}/coins/{crypto_id.lower()}"
            params = {
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false"
            }
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            market_data = data.get("market_data", {})
            return {
                "id": data.get("id"),
                "symbol": data.get("symbol", "").upper(),
                "name": data.get("name"),
                "current_price_usd": market_data.get("current_price", {}).get("usd"),
                "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
                "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
                "price_change_24h": market_data.get("price_change_percentage_24h"),
                "price_change_7d": market_data.get("price_change_percentage_7d"),
                "price_change_30d": market_data.get("price_change_percentage_30d"),
                "ath": market_data.get("ath", {}).get("usd"),
                "ath_date": market_data.get("ath_date", {}).get("usd"),
                "atl": market_data.get("atl", {}).get("usd"),
                "atl_date": market_data.get("atl_date", {}).get("usd"),
                "circulating_supply": market_data.get("circulating_supply"),
                "total_supply": market_data.get("total_supply"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_stock_quote(self, symbol: str) -> dict:
        """Get stock quote data"""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": ALPHA_VANTAGE_KEY
            }
            response = await self.client.get(ALPHA_VANTAGE_API, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Global Quote" not in data or not data["Global Quote"]:
                return {"error": "Stock symbol not found or API limit reached"}
            
            quote = data["Global Quote"]
            return {
                "symbol": quote.get("01. symbol"),
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent"),
                "volume": int(quote.get("06. volume", 0)),
                "latest_trading_day": quote.get("07. latest trading day"),
                "previous_close": float(quote.get("08. previous close", 0)),
                "open": float(quote.get("02. open", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_trending_cryptos(self) -> dict:
        """Get trending cryptocurrencies"""
        try:
            url = f"{CRYPTO_API}/search/trending"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            coins = data.get("coins", [])
            trending = []
            for item in coins[:10]:
                coin = item.get("item", {})
                trending.append({
                    "id": coin.get("id"),
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "price_btc": coin.get("price_btc")
                })
            
            return {
                "trending_coins": trending,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def compare_cryptos(self, crypto_ids: list) -> dict:
        """Compare multiple cryptocurrencies"""
        try:
            ids_str = ",".join([c.lower() for c in crypto_ids])
            url = f"{CRYPTO_API}/simple/price"
            params = {
                "ids": ids_str,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            comparison = []
            for crypto_id in crypto_ids:
                if crypto_id.lower() in data:
                    crypto_data = data[crypto_id.lower()]
                    comparison.append({
                        "id": crypto_id,
                        "price_usd": crypto_data.get("usd"),
                        "24h_change": crypto_data.get("usd_24h_change"),
                        "market_cap": crypto_data.get("usd_market_cap")
                    })
            
            return {
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Initialize data fetcher
fetcher = TradingDataFetcher()


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri=AnyUrl("currency://rates"),
            name="Exchange Rates",
            mimeType="application/json",
            description="Current exchange rates for major currencies"
        ),
        Resource(
            uri=AnyUrl("crypto://trending"),
            name="Trending Cryptocurrencies",
            mimeType="application/json",
            description="Currently trending cryptocurrencies"
        ),
        Resource(
            uri=AnyUrl("trading://guide"),
            name="Trading Guide",
            mimeType="text/plain",
            description="Quick reference guide for using the tracker"
        )
    ]


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a specific resource"""
    uri_str = str(uri)
    
    if uri_str == "currency://rates":
        rates = await fetcher.get_exchange_rate("USD", "EUR")
        return json.dumps(rates, indent=2)
    
    elif uri_str == "crypto://trending":
        trending = await fetcher.get_trending_cryptos()
        return json.dumps(trending, indent=2)
    
    elif uri_str == "trading://guide":
        return """
Currency & Crypto Tracker - Quick Guide

AVAILABLE TOOLS:
1. convert_currency - Convert between any currencies
2. get_crypto_price - Get real-time crypto prices
3. get_crypto_details - Detailed crypto market data
4. get_stock_quote - Stock prices and data
5. get_trending_cryptos - See what's trending
6. compare_cryptos - Compare multiple cryptos

POPULAR CRYPTO IDs:
- bitcoin, ethereum, binancecoin, cardano, solana
- ripple, polkadot, dogecoin, avalanche-2, chainlink

TIPS FOR TRADING:
- Monitor 24h price changes
- Check market cap for stability
- Compare volume for liquidity
- Use trending data for opportunities
"""
    
    raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="convert_currency",
            description="Convert an amount from one currency to another using real-time exchange rates",
            inputSchema={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount to convert"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code (e.g., USD, EUR, GBP)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code (e.g., USD, EUR, GBP)"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        ),
        Tool(
            name="get_crypto_price",
            description="Get current price and 24h change for a cryptocurrency",
            inputSchema={
                "type": "object",
                "properties": {
                    "crypto_id": {
                        "type": "string",
                        "description": "Cryptocurrency ID (e.g., bitcoin, ethereum, cardano)"
                    },
                    "vs_currency": {
                        "type": "string",
                        "description": "Currency to compare against (default: usd)",
                        "default": "usd"
                    }
                },
                "required": ["crypto_id"]
            }
        ),
        Tool(
            name="get_crypto_details",
            description="Get detailed market data for a cryptocurrency including ATH, ATL, supply, and price changes",
            inputSchema={
                "type": "object",
                "properties": {
                    "crypto_id": {
                        "type": "string",
                        "description": "Cryptocurrency ID (e.g., bitcoin, ethereum)"
                    }
                },
                "required": ["crypto_id"]
            }
        ),
        Tool(
            name="get_stock_quote",
            description="Get real-time stock quote including price, change, volume, and daily stats",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_trending_cryptos",
            description="Get list of currently trending cryptocurrencies",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="compare_cryptos",
            description="Compare multiple cryptocurrencies side by side",
            inputSchema={
                "type": "object",
                "properties": {
                    "crypto_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of crypto IDs to compare (e.g., ['bitcoin', 'ethereum', 'cardano'])"
                    }
                },
                "required": ["crypto_ids"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "convert_currency":
        result = await fetcher.convert_currency(
            arguments["amount"],
            arguments["from_currency"],
            arguments["to_currency"]
        )
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_crypto_price":
        result = await fetcher.get_crypto_price(
            arguments["crypto_id"],
            arguments.get("vs_currency", "usd")
        )
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_crypto_details":
        result = await fetcher.get_crypto_market_data(arguments["crypto_id"])
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_stock_quote":
        result = await fetcher.get_stock_quote(arguments["symbol"])
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_trending_cryptos":
        result = await fetcher.get_trending_cryptos()
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "compare_cryptos":
        result = await fetcher.compare_cryptos(arguments["crypto_ids"])
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())