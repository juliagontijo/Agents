from urllib.request import Request, urlopen
from urllib.parse import urlparse
from html.parser import HTMLParser

class PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def read_webpage(url: str) -> dict:
    """Read static HTML; JavaScript-rendered content may be unavailable."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Expected an HTTP or HTTPS URL.")
    request = Request(url, headers={"User-Agent": "MyAgent/0.1"})
    with urlopen(request, timeout=20) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "text/plain"}:
            return {"url": url, "error": "Unsupported page format."}
        raw = response.read(1_000_001)
        encoding = response.headers.get_content_charset() or "utf-8"
        final_url = response.url
    html = raw[:1_000_000].decode(encoding, errors="replace")
    if content_type == "text/html":
        parser = PageTextParser()
        parser.feed(html)
        text = "\n".join(parser.parts)
    else:
        text = html
    return {"url": final_url, "text": text[:24000],
            "truncated": len(raw) > 1_000_000 or len(text) > 24000}

def search_web(query: str) -> dict:
    from ddgs import DDGS

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a nonempty string.")

    results = DDGS().text(
        query,
        backend="auto",
        max_results=5,
    )

    return {
        "results": [
            {
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body"),
            }
            for result in results
        ]
    }


search_tools = [
    {
        "type": "function",
        "name": "search_web",
        "description": "Search the web for relevant information.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_webpage",
        "description": "Read a search result page to extract facts absent from its snippet. Returns static page text; may omit dynamic content.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    }
]


