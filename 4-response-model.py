from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAIChat
from agno.models.openai.like import OpenAILike
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field
import httpx
import os

if os.environ.get("ANTHROPIC_API_KEY", "0") != "0":
  model=Claude(id="claude-3-5-sonnet-20240620")
elif os.environ.get("OPENAI_API_KEY", "0") != "0":
  model=OpenAIChat(id="GPT-4o")
elif os.environ.get("OPENAI_LIKE", "0") != "0":
  model=OpenAILike(api_key=os.getenv("OPENAI_LIKE"),
		id="c4dt",
		base_url="http://localhost:3001/api/v1/openai")
else:
  model=LMStudio()

def get_person(email: str) -> str:
    """This function returns the description of this person in the EPFL database.
    It returns the full description, without any differentiation.

    Args:
        email (str): the email of the person

    Returns:
        str: json of the person"""
    url = f"https://search-api.epfl.ch/api/ldap?q={email}&hl=en"
    response = httpx.get(url)
    response.raise_for_status()
    return response.text

def get_unit(name: str) -> str:
    """This function returns the description of this unit in the EPFL database.
    It returns the full description, without any differentiation.
    For some of the units, the name needs to end in "-GE" to get the list of employees. 

    Args:
        name (str): the name of the unit

    Returns:
        str: json of the unit, including people"""

    url = f"https://search-api.epfl.ch/api/unit?q={name}&hl=en"
    response = httpx.get(url)
    response.raise_for_status()
    return response.text

class RSEDescription(BaseModel):
    organization: str = Field(..., description="Abbreviation of the rganization this unit is in - a school, college, or vice-presidency.")
    unit_name: str = Field(..., description="Abbreviation of the unit or the lab.")
    unit_head: str = Field(..., description="Head of the unit.")
    person_name: str = Field(..., description="Name of the person.")
    person_email: str = Field(..., description="Email of the person.")
    position: str = Field(..., description="Position of the person.")
    page_url: str = Field(..., description="URL of the person's page.")
    
    def csv(self) -> str:
        """Generate a CSV representation of the RSEDescription instance."""
        return f"{self.organization},{self.unit_name},{self.unit_head},{self.person_name},{self.position},{self.page_url}"

    
class RSEs(BaseModel):
    rses: list[RSEDescription]

agent = Agent(
    model=model,
    description="Retrieve RSEs from the EPFL database",
    tools=[get_person, get_unit],
    use_json_mode=True,
    instructions=dedent("""\
        Your goal is to help me make a list of all Research Software Engineers (RSE) and their responsible in the 
        labs, centers, and other units of EPFL.
        You will receive one email at the time of an RSE, and then have to search their
        unit, their responsible, and eventual other RSEs in the same unit.
        It is important to search for the other RSEs in the same unit, as I don't have
        all the emails of all RSEs.
        RSEs can have different names: Research software engineers, computer scientists, software engineers, etc.
        A Doctoral Assistant is NOT an RSE, so don't include them.
        An exception is if the Doctoral Assistant is the email given to you in the input.
        In this case, also include the Doctoral Assistant in the output.

        The organization is one of EPFL schools, colleges, or Vice Presidencies.
        Use the abbreviation of the organization, as given in the path of the person.
        The unit name is the name of the team or lab the person belongs to.
        The unit head is the name of the head of the team or lab.
        The person name is the name of the person.
        The position is the title of the person.
        The page url is the URL of the person's page.

        To search all this information, you can use the tools I provided.
        You can use the get_person tool to get the person's page.
        You can use the get_unit tool to get the unit's page.
        If a person has more than one unit, look for the unit which looks like a lab
        or a team, and use that one.
        A lot of the centers or units, which are not a lab, end in -GE for the actual
        unit.
        
        In your final response, do not reflect on your thought process, but only
        return the final answer as JSON, as described.
    """),
    response_model=RSEs,
    # show_tool_calls=True,
    # markdown=True,
    # debug_mode=True
)

with open("emails.txt", "r") as file:
    emails = file.readlines()

for email in emails[:4]: 
    email = email.strip()
    print(email)
    run: RunResponse = agent.run(email)
    if isinstance(run.content, RSEs):
        for rse in run.content.rses:
            print(rse.csv())
