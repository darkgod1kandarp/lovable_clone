from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
import asyncio

# Storing the config
config = CrawlerRunConfig(
    deep_crawl_strategy=BFSDeepCrawlStrategy(
        max_depth=2,        # How many levels of links to follow
        max_pages=10,       # Max pages to crawl total
        include_external=False  # Stay on the same domain only
    )
)

def extract_website_from_prompt(user_prompt: str) -> str:
    # Simple heuristic to extract a URL from the user prompt
    import re
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, user_prompt)
    if match:
        return match.group(0)
    else:
        return "https://www.linkedin.com/"  # Default URL if none found
    
# Function to run the crawl
async def crawl_website(prmopt:str) -> list:
    """Example of running a crawl with the AsyncWebCrawler and BFSDeepCrawlStrategy.
    """
    url = extract_website_from_prompt(prmopt)
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=url, config=config)
        print(f"Crawl Results: {results}")
        
    return results

# if __name__ == "__main__": 
#    asyncio.run(crawl_website())
