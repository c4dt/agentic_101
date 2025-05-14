from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAIChat
from agno.models.openai.like import OpenAILike
from agno.models.anthropic import Claude
from agno.tools.firecrawl import FirecrawlTools
from pydantic import BaseModel, Field
from markdownify import markdownify as md
import httpx
import os

if os.environ.get("ANTHROPIC_API_KEY", "0") != "0":
  model=Claude(id="claude-3-7-sonnet-latest")
elif os.environ.get("OPENAI_API_KEY", "0") != "0":
  model=OpenAIChat(id="gpt-4.1")
elif os.environ.get("OPENAI_LIKE", "0") != "0":
  model=OpenAILike(api_key=os.getenv("OPENAI_LIKE"),
		id="c4dt",
		base_url="http://localhost:3001/api/v1/openai")
else:
  model=LMStudio()

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
    if not url.startswith("https://"):
        url = f"https://{url}"
    
    print(f"Fetching URL: {url}")
    response = httpx.get(url)
    response.raise_for_status()
    return md(response.text)

def add_news(agent: Agent, news: NewsSummary):
    """Adds a new summary to the list of news"""
    print(f"Adding news {news}")
    agent.session_state["news_list"].append(news)


list_news = Agent(
    model = model,
    description = "Fetches the latest news and returns a summary",
    tools=[get_url, add_news],
    use_json_mode=True,
    add_state_in_messages=True,
    session_state={"personality": "", "news_list": []},
    instructions=dedent("""\
        Visit the url from the prompt, and then add all news articles related to digital trust to the news_list.
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
    model = model,
    description= "Returns the top 3 articles by relevance",
    use_json_mode=True,
    add_state_in_messages=True,
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
    url: str = Field(..., description="The URL to the article")
    description: str = Field(..., description="The paragraph for this weekly pick")

write_weekly = Agent(
    model = model,
    description="Write a weekly pick for the article",
    tools=[FirecrawlTools(crawl=False)],
    use_json_mode=True,
    add_state_in_messages=True,
    session_state={"personal_interest": ""},
    instructions= dedent("""\
        For the URL in the prompt, fetch the website, and create a weekly pick.
        A weekly pick has the following format:
        - it is a 1-paragraph, about 500 characters personal opinion
        - take into account the {personal_interest} of the requester
        - it puts digital trust in the foreground
        - it should talk about how the problem might be solved
        
        It should have a personal note and not only be a summary of the article.
        Write in a nice style, not too formal. Use short sentences, and avoid too complicated words.
        Don't add adverbs and adjectives all over the place.
        
        For the final reply, only send the JSON, nothing else. Don't introduce the JSON, just send the json.
        """),
    response_model=WeeklyPick
)

list_news.session_state={"personality": dedent("""\
    I am a passionate programmer, software engineer, doing everything from javascript to python, but mainly rust.
    Everything digital trust is very important to me, mostly with regard to how the governments are the best
    protection for the people to protect them against the big corporations with regard to
    privacy, trust, and even security."""), "news_list": []}

# for url in ["lobste.rs"]:
for url in ["news.ycombinator.com", "arstechnica.com", "www.404media.co", "lobste.rs"]:
    list_news.run(url)

order_news.session_state = list_news.session_state
# order_news.session_state["news_list"] = [NewsSummary(url='https://blog.trailofbits.com/2025/05/14/the-cryptography-behind-passkeys/', summary='An in-depth examination of the cryptographic mechanisms underlying passkeys and the WebAuthn standard, explaining how they improve digital authentication. The article covers how passkeys use asymmetric cryptography to replace passwords, why phishing is mitigated by origin binding, and details of how browsers, authenticators, and extensions interact to deliver strong security. The piece provides actionable recommendations for users and developers, describes attack surface considerations, and suggests future improvements and extensions for security. The role of governments in policy and enforcement is briefly mentioned regarding platform and recovery risks.', dt_relevance=5.0, personal_relevance=5.0), NewsSummary(url='https://blog.stillgreenmoss.net/sms-2fa-is-not-just-insecure-its-also-hostile-to-mountain-people', summary='This article critiques the widespread use of SMS-based two-factor authentication (2FA), describing both its insecurity and lack of accessibility in rural regions. Technical and infrastructural issues prevent people from receiving authentication codes, especially over Wifi or in areas with poor cell service. The story highlights that alternatives like TOTP (app-based 2FA) are not much more user-friendly for many, leaving some people locked out of essential accounts. The author calls for a more inclusive and secure approach to digital authentication, emphasizing that governments and industry should do more to support the privacy, safety, and accessibility of all users.', dt_relevance=4.0, personal_relevance=4.0), NewsSummary(url='https://nextcloud.com/blog/nextcloud-android-file-upload-issue-google/', summary="Nextcloud details how Google revoked access to key Android file system APIs, crippling full file uploads for their app and impacting user control over data and privacy. The post accuses Google of anticompetitive behavior under the guise of 'security,' granting its own apps privileges that others lack. The issue raises larger concerns about digital trust, gatekeeping, and regulatory inaction, as small players like Nextcloud are left with few remedies. The blog advocates for stronger policy enforcement and greater digital sovereignty, specifically calling for governments to address Big Tech's gatekeeping practices to protect users’ privacy and autonomy.", dt_relevance=4.0, personal_relevance=5.0), NewsSummary(url='https://comsec.ethz.ch/research/microarch/branch-privilege-injection/', summary="Branch Privilege Injection is a new class of vulnerability that exploits race conditions in Intel CPUs' branch predictors, bypassing hardware mitigations for Spectre-BTI. The attack enables privileged memory leaks across security boundaries despite eIBRS and IBPB mitigations, affecting all Intel CPUs since the 9th gen. Mitigations require BIOS and software updates. The article calls for prompt OS and firmware updates, highlights limits of current hardware security, and underlines the ongoing arms race in hardware/software defenses versus advanced attacks.", dt_relevance=5.0, personal_relevance=5.0), NewsSummary(url='https://arstechnica.com/security/2025/05/google-introduces-advanced-protection-mode-for-its-most-at-risk-android-users/', summary="Google has launched 'Advanced Protection' for Android 16. It targets at-risk users like journalists and government officials, enabling various deep security measures (blocking insecure networks, memory safety, intrusion logging, etc.) with one setting. It addresses the growing threat of sophisticated, targeted hacking campaigns. Similar to Apple's Lockdown Mode, it improves trust and privacy for high-target individuals and by extension, all users.", dt_relevance=10.0, personal_relevance=10.0), NewsSummary(url='https://arstechnica.com/security/2025/05/ai-agents-that-autonomously-trade-cryptocurrency-arent-ready-for-prime-time/', summary="Researchers demonstrate a 'context manipulation' attack on ElizaOS, an open-source AI agent for blockchain-based transactions. The attack uses prompt injection to plant false memory events, causing the bot to send crypto to an attacker's wallet even when instructed by legitimate users. This highlights serious digital trust and security flaws in AI systems with persistent memory, especially for financial or autonomous use, and demonstrates the urgent need for integrity checks and defensive controls in such agents.", dt_relevance=9.5, personal_relevance=9.0), NewsSummary(url='https://arstechnica.com/tech-policy/2025/05/meta-is-making-users-who-opted-out-of-ai-training-opt-out-again-watchdog-says/', summary="EU privacy watchdog Noyb issued a cease-and-desist letter to Meta, arguing the company forces EU users to re-opt-out of AI training—potentially violating GDPR and eroding digital trust. Users who previously objected to data use for AI training must do so again or their data will be included permanently, raising concerns over compliance, user rights, and trust in Big Tech's privacy promises. Meta maintains it follows EU guidelines but critics call the move deceptive and against data protection principles.", dt_relevance=8.5, personal_relevance=9.0), NewsSummary(url='https://arstechnica.com/gadgets/2025/05/nextcloud-accuses-google-of-big-tech-gatekeeping-over-android-app-permissions/', summary="Nextcloud claims Google's Play Store rules restrict full file access for independent apps, citing a refusal to grant 'All files access' permission for its Android client. This impacts users' ability to upload/sync non-media files to personal clouds, illustrating Big Tech's control over user data, competition, and digital sovereignty. The issue raises important digital trust questions for users and smaller developers relying on platforms dominated by Google.", dt_relevance=7.5, personal_relevance=8.0), NewsSummary(url='https://arstechnica.com/ai/2025/05/welcome-to-the-age-of-paranoia-as-deepfakes-and-scams-abound/', summary='AI-driven deepfakes and scams are making digital verification more necessary and complex, causing people to scrutinize even routine online interactions. The proliferation of fake personas and fraudulent calls is fueling a climate of distrust, with businesses and individuals adopting new, sometimes burdensome, verification and authentication methods. While solutions are emerging (AI deepfake detectors, biometric verification), improving digital trust may require both technological and policy responses.', dt_relevance=8.0, personal_relevance=8.0), NewsSummary(url='https://arstechnica.com/ai/2025/05/gop-sneaks-decade-long-ai-regulation-ban-into-spending-bill/', summary='House Republicans have added a provision to the US Budget Reconciliation bill that would prohibit state/local AI regulation for 10 years. This measure could block state-level privacy, bias, and AI data transparency laws, effectively weakening government oversight and protections against corporate abuses—an issue central to digital trust, especially for those who believe government checks are critical against emerging AI risks and big tech overreach.', dt_relevance=9.0, personal_relevance=9.5), NewsSummary(url='https://arstechnica.com/gadgets/2025/05/vpn-firm-says-it-didnt-know-customers-had-lifetime-subscriptions-cancels-them/', summary='After being acquired, VPNSecure abruptly canceled thousands of lifetime subscriptions, citing unawareness of these deals. Customers expressed outrage, highlighting issues of transparency, reliability, and consumer trust in privacy/security products. The move undermines confidence in digital service providers’ commitments, especially for VPNs that form a foundation of personal digital privacy and trust.', dt_relevance=7.0, personal_relevance=7.5), NewsSummary(url='https://www.404media.co/license-plate-reader-company-flock-is-building-a-massive-people-lookup-tool-leak-shows/', summary="Flock Safety, known for its ALPR (automatic license plate reader) cameras throughout the U.S., is reportedly developing 'Nova,' a tool that will allow law enforcement to cross-reference license plate data with personal information obtained from people lookup tools, data brokers, and sources in breached data. According to internal presentations and messages, Nova aims to help police 'jump from LPR to person,' enabling the tracking and identification of people nationwide without requiring warrants or court orders. This has raised major ethical concerns among employees regarding the inclusion of hacked/breached data. The tool is already in early use by some law enforcement agencies.", dt_relevance=9.5, personal_relevance=10.0), NewsSummary(url='https://www.404media.co/republicans-try-to-cram-ban-on-ai-regulation-into-budget-reconciliation-bill/', summary='A new provision added to the Budget Reconciliation bill by House Republicans would prohibit all U.S. states from regulating artificial intelligence for 10 years. The proposed language would cover both new generative AI tools and older automated systems, making it impossible to enforce state-level protections, disclosure requirements, or bias audits related to AI. This would effectively block recently passed regulations in states like California and New York and prevent future local efforts to protect citizens’ privacy, safety, and rights in the age of rapidly advancing AI.', dt_relevance=8.5, personal_relevance=9.0), NewsSummary(url='https://blog.stillgreenmoss.net/sms-2fa-is-not-just-insecure-its-also-hostile-to-mountain-people', summary='A personal account details the practical and security failures of SMS-based two-factor authentication (2FA), especially for rural populations lacking reliable cell service. The article points out systemic shortcomings, the obstacles to switching to more secure methods such as TOTP, and the wider impact of inaccessible corporate security choices. It advocates for more robust, government-driven regulations to ensure accessible, secure authentication for all users.', dt_relevance=9.5, personal_relevance=9.5)]
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
# ordered.content.url_list = [Url(url='https://blog.trailofbits.com/2025/05/14/the-cryptography-behind-passkeys/'), Url(url='https://comsec.ethz.ch/research/microarch/branch-privilege-injection/'), Url(url='https://googleprojectzero.blogspot.com/2025/05/breaking-sound-barrier-part-i-fuzzing.html')]

print("Ordered list of URLs:", ordered.content.url_list)
write_weekly.session_state = order_news.session_state
for article in ordered.content.url_list:
    wp: RunResponse = write_weekly.run(article.url)
    if isinstance(wp.content, WeeklyPick):
        print(f"My take: \"{wp.content.description}\" - {wp.content.url}")
    else:
        print("Oups - weekply pick failed:", wp.content)
