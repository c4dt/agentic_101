from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from markdownify import markdownify as md
import httpx
import os

def get_url(url: str) -> str:
    """This function returns the webpage of the given URL.
    It returns the full description, without any differentiation.

    Args:
        url (str): the URL of the page

    Returns:
        str: the webpage as a markdown"""
    print(f"Fetching URL: {url}")
    response = httpx.get(url)
    response.raise_for_status()
    return md(response.text)

if os.environ.get("ANTHROPIC_API_KEY", "0") != "0":
  model=Claude(id="claude-3-5-sonnet-20240620")
elif os.environ.get("OPENAI_API_KEY", "0") != "0":
  model=OpenAIChat(id="GPT-4o")
else:
  model=LMStudio()

agent = Agent(
    model=model,
    description="Retrieve RSEs from the EPFL database",
    tools=[get_url],
    instructions=dedent("""\
        Your goal is to help me make a list of all Research Software Engineers (RSE) and their responsible in the 
        labs, centers, and other units of EPFL.
        You will receive one email at the time of an RSE, and then have to search their
        unit, their responsible, and eventual other RSEs in the same unit.
        It is important to search for the other RSEs in the same unit, as I don't have
        all the emails of all RSEs.
        RSEs can have different names: Research software engineers, computer scientists, software engineers, etc.
        A Doctoral Assistant is NOT an RSE, so don't include them.
        For each RSE, you need to output one line of a CSV:

        The CSV data should contain the following fields:
        - organization, unit name, unit head, person name, position, page url

        The organization is one of EPFL schools, colleges, or Vice Presidencies.
        The unit name is the name of the team or lab the person belongs to.
        The unit head is the name of the head of the team or lab.
        The person name is the name of the person.
        The position is the title of the person.
        The page url is the URL of the person's page.

        To search all this information, you can use two search pages from EPFL:

        - for information about a person https://people.epfl.ch/{first.last}
        - for information about a unit https://search.epfl.ch/?filter=unit&q={unit_name}

        If a person has more than one unit, look for the unit which looks like a lab
        or a team, and use that one.
        A lot of the centers or units, which are not a lab, end in -GE for the actual
        unit.

        Start the output with the text "== OUTPUT ==" on an empty line.
        The output should only be the lines of the CSV, without any other text.
        Each RSE you find should be on a separate line.
        Do not add the header of the CSV.
        Do not add any explanations.
        I just need the lines of the CSV representing one RSE per line.
        Thank you very much for your collaboration!
    """),
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

with open("emails.txt", "r") as file:
    emails = file.readlines()

for email in emails[0:1]:
    email = email.strip()
    if email:  # Skip empty lines
        print(email)
        run: RunResponse = agent.run(email)
        output_parts = run.content.split("== OUTPUT ==")
        if len(output_parts) == 2:
            print(output_parts[1].strip())
