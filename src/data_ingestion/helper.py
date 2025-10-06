# src/data_ingestion/helper.py
import time 
import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEBLINKS_PATH = PROJECT_ROOT / "src" / "model_training" / "weblinks.json"
NYSE_URL = "https://www.nyse.com/listings_directory/stock"

async def main():
    """
    Scrapes all stock symbols from the NYSE listings directory and updates weblinks.json.
    """
    print("--- Starting NYSE Ticker Scraping ---")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        print(f"Navigating to {NYSE_URL}...")
        await page.goto(NYSE_URL, wait_until="networkidle")
        time.sleep(1)

        scraped_tickers = set()
        page_num = 1

        while True:
            print(f"Scraping page {page_num}...")
            time.sleep(1)
            # Wait for the table to be visible
            await page.wait_for_selector("table.table-data")
            time.sleep(1)
            # Get all symbol elements from the first column of the table
            symbol_elements = await page.query_selector_all("table.table-data tbody tr td:first-child a")
            time.sleep(1)
            for element in symbol_elements:
                time.sleep(1)
                ticker = await element.text_content()
                if ticker:
                    scraped_tickers.add(ticker.strip())
            time.sleep(1)
            # Find the "Next" button
            next_button = page.locator('a:has-text("Next ›")')
            
            # Check if the "Next" button is visible and enabled
            if await next_button.is_visible() and await next_button.is_enabled():
                await next_button.click()
                time.sleep(1)
                # Wait for the page to navigate and the network to be idle
                await page.wait_for_load_state("networkidle")
                page_num += 1
                time.sleep(1)
            else:
                print("No more 'Next' pages found. Scraping complete.")
                break
        
        await browser.close()

        if not scraped_tickers:
            print("Warning: No tickers were scraped. Aborting file update.")
            return

        print(f"\nSuccessfully scraped {len(scraped_tickers)} unique tickers.")
        
        # --- Update weblinks.json ---
        print(f"Updating {WEBLINKS_PATH}...")
        try:
            with open(WEBLINKS_PATH, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty, create a new structure
            data = {"tickers": []}
            
        # Update the list with the new tickers (sorted alphabetically)
        data['tickers'] = sorted(list(scraped_tickers))
        
        with open(WEBLINKS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
            
        print("--- Update Complete ---")

if __name__ == "__main__":
    asyncio.run(main())