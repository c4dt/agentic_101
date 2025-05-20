from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.tools.firecrawl import FirecrawlTools
from pydantic import BaseModel, Field
from markdownify import markdownify as md
from common import get_url_cached, model

# Change the personality of the weekly-pick writer to match your interests.
personality = dedent("""\
    I am a passionate programmer, software engineer, doing everything from javascript to python, but mainly rust.
    Everything digital trust is very important to me, mostly with regard to how the governments are the best
    protection for the people to protect them against the big corporations with regard to
    privacy, trust, and even security.""")

# Global configuration for all agents
AGENT_CONFIG = {
    "model": model,
    "use_json_mode": True,
    "add_state_in_messages": True,
    "show_tool_calls":True,
    # "debug_mode":True
}

class NewsSummary(BaseModel):
    url: str = Field(..., description="The URL to the article")
    summary: str = Field(..., description="A short summary of the article")
    dt_relevance: float = Field(..., description="Relevance to digital trust and cybersecurity. 0 = none, 10 = full relevance")
    personal_relevance: float = Field(..., description="Personal relevance with regard to the interest of the querier. 0 = none, 10 = full relevance")

def get_url(url: str) -> str:
    """This function returns the webpage of the given URL.
    It returns the full description, without any differentiation.

    Args:
        url (str): the URL of the page

    Returns:
        str: the webpage as a markdown"""
    return md(get_url_cached(url))

def add_news(agent: Agent, news: NewsSummary):
    """Adds a new summary to the list of news"""
    print(f"Adding news {news}")
    agent.session_state["news_list"].append(news)

list_news = Agent(
    **AGENT_CONFIG,
    description="Fetches the latest news and returns a summary",
    tools=[get_url, add_news],
    session_state={"personality": "", "news_list": []},
    instructions=dedent("""\
        Visit the url from the prompt, and then choose the 5 most relevant articles related to digital trust to the news_list.
        For the dt_relevance field, only consider the relevance with regard to digital trust, cybersecurity, policy,
        attacks, as well as defenses. Consider articles which talk about defenses or how to fix
        privacy issues higher than articles which only complain about those issues.
        For the personal_relevance, consider the personality of the querier: {personality}.
        """),
)

class Url(BaseModel):
    url: str = Field(..., description="The URL to the article")
    
class UrlList(BaseModel):
    url_list: list[Url]

order_news = Agent(
    **AGENT_CONFIG,
    description= "Returns the top 3 articles by relevance",
    instructions=dedent("""\
        Here is a list of articles scraped by a previous agent:
        
        {news_list}
        
        You need to return the top 3 articles from this list.
        Use the dt_relevance and personal_relevance fields which go from 0 (not relevant) to 10 (very relevant).
        Of course you should also use the summary field.

        For the final reply, only send the JSON, nothing else. Don't introduce the JSON, just send the json.
        """),
    response_model=UrlList,
)

class WeeklyPick(BaseModel):
    # COMPLETE the WeeklyPick class

write_weekly = Agent(
    **AGENT_CONFIG,
    description="Write a weekly pick for the article",
    tools=[FirecrawlTools(crawl=False)],
    session_state={"personal_interest": ""},
    instructions= dedent("""\
        COMPLETE INSTRUCTIONS HERE
        """),
    response_model=WeeklyPick
)

list_news.session_state={"personality": personality, "news_list": []}

for url in ["www.404media.co"]:
    list_news.run(url)

order_news.session_state = list_news.session_state
# order_news.session_state["news_list"] = []
print("List of news articles:", order_news.session_state["news_list"])

ordered: RunResponse = order_news.run("follow the instructions")
if not isinstance(ordered.content, UrlList):
    print("Oups - something went wrong with the content:", ordered.content)
    exit(1)

# class RRUL:
#     url_list: list[Url]
# class RR:
#     content: RRUL = RRUL()
# ordered = RR()
# ordered.content.url_list = []

print("Ordered list of URLs:", ordered.content.url_list)
write_weekly.session_state = order_news.session_state
for article in ordered.content.url_list:
    wp: RunResponse = write_weekly.run(article.url)
    if isinstance(wp.content, WeeklyPick):
        print(f"My take: \"{wp.content.description}\" - {wp.content.url}")
    else:
        print("Oups - weekply pick failed:", wp.content)
