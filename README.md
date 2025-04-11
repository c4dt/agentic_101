# Using Agentic LLM to Extract Emails

In this article you'll learn how to use agentic LLMs with the [Agno](https://agno.com) framework,
and how the solution for this use-case compares to a non-agentic Python script.
I suppose you have moderate knowledge of Python, HTML, and Javascript.
Jump to the part of the article you're most interested in, or continue reading.

- [Use-case](#the-use-case) is to find all Research Software Engineers at EPFL, starting from a partial
list of emails
- [Frameworks](#langchain-and-agno) considered: Langchain (TLDR) and Agno (Chosen)
- [Implementations](#the-implementations) of the different apporoaches I took
  - [Naive apporach](#the-naive-approach) where the LLM needs to do everything magically correct
  - [Create a website downloader](#tool-creation---website-downloader) which is very easy with Agno
  - [Use EPFL API](#use-the-epfl-api-for-persons-and-units) which gives first results
  - [Add response models](#add-response-models) to format the answers
  - [No LLMs approach](#look-ma---no-llms) with a non-agentic python script
- [Running the code](#running-the-code) to see for yourself
- [Conclusion](#agents-or-not-agents-that-is-the-question) of my first encounter with LLM Agents

## The Use-Case

With other [Research Software Engineers](https://rse.epfl.ch) (RSE) at EPFL, 
we put together a survey to better understand how RSEs in EPFL work.
Instead of sending the survey to all@epfl.ch, I
wanted to target more specifically RSEs, also to avoid spamming everybody at EPFL.
As we have a mailing list with 30+ RSEs, I thought this is a good use-case for an agentic LLM:

1. start with an email on the list
2. look it up on the EPFL webpage
3. click the link to their unit to get a list of all employees
4. collect all RSE-like job titles and the emails from this page

Sounds like one of these tasks where you can spend hours clicking on links, copy/pasting stuff,
and this is where an agent should help, no? Turns out yes, even if the non-agentic way for this 
specific use-case is shorter - see at the end.

## Langchain and Agno

There are many frameworks which promote the use of agentic LLMs, one of the most notable is
[Langchain](https://python.langchain.com/docs/introduction/). I spent nearly an hour trying
to understand if I need LangChain, LangGraph, LangGraph Platform, or LangSmith. Spending some
time with the examples made me think that there must be an easier way. Way too many parameters 
had to be defined by the user, connections were not clear. It looked like one of these
libraries which grew organically. Which is nice if you saw them growing. But not nice if you're
coming late to the party...

After some Googling and looking at examples I settled on [Agno](https://agno.com). It looked
perfect to me: simple examples, with hello world being just a couple of lines. Good documentation,
even if they use an LLM in their search bar. It has a LOT of model adapters, even for 
[LM Studio](https://lmstudio.ai/), which I like on my Mac M2 with 64GB of RAM.

# The Implementations

I did four agentic implementations: 

1. [Naive apporach](#the-naive-approach) where the LLM needs to do everything magically correct,
whith only a little bit of information and guidance. This mostly failed because of the tools in
Agno for scraping websites, which failed on EPFL's pages.
2. [Create a website downloader](#tool-creation---website-downloader) replaces Agno's tools with
a handwritten downloader, which is very simple and very effective. Unfortunately, this made it clear
that EPFL's pages don't work without javascript.
3. [Use EPFL API](#use-the-epfl-api-for-persons-and-units) adds two other tools to the agent, which
query directly EPFL's API and thus have nicely formatted results. This gave the first useful results
of the LLM agents.
4. [Add response models](#add-response-models) to make the output more reproducible and nicer. Here I
also refined the instruction sent to the agent to get better results.

The fith implementation, [5-normal.py](./5-normal.py) is a non-agentic implementation, which took me
way less time than all the other combined :)

## The Naive Approach

You can find the full source code of this approach here: [1-naive.py](./1-naive.py).
I will only show the most relevant parts of this approach to give an introduction to Agno.

Here is the first code I used to create an agent with Agno:

```python
from textwrap import dedent
from agno.tools.website import WebsiteTools
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    model=Claude(id="claude-3-5-sonnet-20240620"),
    tools=[WebsiteTools()],
    instructions=dedent("""
    [... Lots of instructions here ...]
    """)
)

print(agent.run("linus.gasser@epfl.ch").content)
```

I really like the minimalistic approach of Agno here: every line just gives the absolute necessary
information, no superfluous commands are needed.
Here is a short breakdown of the elements of the `Agent`:

- `model` takes a [lot of models](https://docs.agno.com/models/introduction): Anthropic, OpenAI, and even 
LM Studio and Ollama for models running on your local computer!
- `tools` allow the LLM to interact with the outside world, Agno has 
[a lot of toolkits](https://docs.agno.com/tools/toolkits/toolkits)
- `instructions` direct the model in its task and is written in plain English

All the magic happens in the `Lots of instructions here`, which are instructions written in common
English, which is both good and bad:

- The good
  - You don't need to know any programming language
  - It's like describing the task to another human
- The bad
  - Without some programming knowlege the output will be unusable
  - The execution is not deterministic - sometimes it works, sometimes it doesn't

### How to talk to an agent

The following is the text which explains the agent how it can find the necessary information.
It is written in plain English, but without programming knowledge it's difficult
to come up with a text like this:

```md
To search all this information, you can use two search pages from EPFL:

- for information about a person https://people.epfl.ch/{first.last}
- for information about a unit https://search.epfl.ch/?filter=unit&q={unit_name}
```

Agno takes care to describe the `WebsiteTools` to the LLM, so it knows how to request
a webpage.

You can look at the rest of the text where I describe the task to the agent.
Some of the explanations are repetitive, because writing them down only once
didn't give consistent results...

### Tools for LLM Agents

This first version mostly failed because the Web-tools provided by Agno were not useful:
- `Crawl4aiTools` installed its own browser, but then produced errors when using it
- `WebsiteTools` crawled a lot of additional webpages, slowing down the system

So in the next step I created my own website downloader!

## Tool Creation - Website Downloader

You can find the full source code of this 2nd approach here: [2-curl.py](./2-curl.py).
I will only show the most relevant parts of this approach to give an introduction to Agno.

Agno has a very simple way of adding your own tools: you write a python
method, add a description which will be passed to the agent, and you're done!
The first tool to get a webpage looked like this (thanks Claude for writing that...):

```python
def get_url(url: str) -> str:
    """ [Description]"""
    response = httpx.get(url)
    return response.text
```

And adding it to the agent is a single line in the Agent setup:

```python
agent = Agent(
  [...]
  tools=[get_url],
```

Agno will then include the description of the method, its name, and how to call it,
directly to the agent.
Most of the time the agent will correctly pick this up.
Sometimes it will not, which is frustrating when you try to debug it.

### LLM Agent vs. HTML code

While debugging to understand why the LLM couldn't use this output, I realized that
EPFL's results at https://people.epfl.ch are nice to look at as a human, but
very ugly behind the scene.
Here are some of the lines returned to the agent:

```html
			<div class="d-flex flex-wrap justify-content-between align-items-baseline"><!-- edit profile -->
				<h1 id="name" class="mr-3 pnX">Linus Gasser</h1>
					<div>
						<a href="/linus.gasser/edit?lang=en">Edit profile</a>
					</div>
			</div><!-- edit profile -->
			<div class="row"><!-- ROW1 -->
			<div class="col-md-12"><!-- ROW1,single -->
			<div class="row"><!-- ROW2 -->
			<div class="col-md-6"><!-- ROW2,L -->
				<div class="row"> <!-- ROW3 -->
				<div class="col-sm-12 col-md-6 mb-3">
							<img src="/private/common/photos/links/111443.jpg?ts=1744366852" alt="photo placeholder image" class="img-fluid" >
	 			</div>
	 			</div> <!-- ROW3 -->
	 			<div class="row"> <!-- ROW4 -->
		 			<div class="col-sm-12"><!--	CONTACT	-->
			 			<h4>Research Software Engineer</h4>
						<p>
```

This looks very cryptic, not only to you and me, but also to an LLM agent.
Googling for `python convert html to markdown` returned a tool which takes HTML code as
input, and returns a more simple representation in markdown.
The updated version looks like this:

```python
from markdownify import markdownify as md
[...]
def get_url(url: str) -> str:
  [...]
  return md(response.text)
```

That was not very complicated, but allowed the agent to actually understand the
webpage.

### LLM Agent vs. Javascript

The next problem was the following, which is the output of the agent when requesting
the description of a unit from EPFL's webpage:

```text
**We're sorry but EPFL Units doesn't work properly without JavaScript enabled. Please enable it to continue.**                                                         
```

This is bad.
It means that the search result of the unit is empty if no javascript is enabled.
Enabling Javascript with our tool is not for the faint of the heart, 
as a search for `python run javascript html`
shows: you need to install a lot of additional things to get it to work.
Or use the `Crawl4aiTools` tool which runs a hidden webbrowser, but which fails to run
on my computer :( 

## Use the EPFL API for Persons and Units

You can find the full source code of this 3rd approach here: [3-epfl-api.py](./3-epfl-api.py).

I spent the next hour looking for APIs from the EPFL which allow to search for persons
and units in a more programming-friendly way.
In the end I found two APIs:

- [EPFL People API](https://www.npmjs.com/package/epfl-people-api)
- [EPFL Unit API](https://www.npmjs.com/package/epfl-unit-api)

Extracting the correct URLs from these libraries took me longer than I want to admit
as a seasoned programmer.
Where are LLMs when we need them most?
So now, instead of relying on webpages, we can use the API.
This allows our tool to send a specific query, and the server replies with a nicely formatted
answer, for example:

```bash
% curl 'https://search-api.epfl.ch/api/ldap?q=linus.gasser@epfl.ch&hl=en'
[{"sciper":"111443","rank":0,"name":"Gasser","firstname":"Linus","email":"linus.gasser@epfl.ch","accreds":[{"phoneList":["+41216936770"],"officeList":["BC 047"],"path":"EPFL/IC/C4DT/C4DT-GE","acronym":"C4DT-GE","name":"C4DT - Administration","order":1,"position":"Research Software Engineer","rank":0,"code":13613}],"profile":"linus.gasser"}]
```

This is something easily understandable by programs and LLMs, has always the same format,
and doesn't require javascript.

### API Tools

The new python tool methods look very similar to the `get_url` one, as it is also a simple
GET request:

```python
def get_person(email: str) -> str:
    """[Description]"""
    url = f"https://search-api.epfl.ch/api/ldap?q={email}&hl=en"
    response = httpx.get(url)
    return response.text

def get_unit(name: str) -> str:
    """[Description]"""
    url = f"https://search-api.epfl.ch/api/unit?q={name}&hl=en"
    response = httpx.get(url)
    return response.text
```

And adding them to the agent, like before:

```python
agent = Agent(
  [...]
    tools=[get_person, get_unit],
```

I also described the tools to use in the `instructions` part of the `Agent`, in addition
to the description of the methods Agno passes to the agent.
This allows the Agent to interact with the outside world using these two tools.
That's all there is to it!

I also tried to return the JSON object directly by calling `return response.json()`, but Agno
didn't like having a JSON object.
So now it gets the string representation of the JSON, which is nice!

### Results

Using the two tools which connect directly to the EPFL API finally gives some 
useful results.
But there are still some issues:

- like in programming, some of the instructions are ambivalent. For example:
`A Doctoral Assistant is NOT an RSE, so don't include them.` - but what if the
email address from the mailing list is a doctoral assistant?
In that case I would still like the person to show up in the results!
- unless in programming, the results are not deterministic: from one run
to another, the results can, and will, differ
- the results also depend a lot on the LLM chosen: some models are better suited
than others.
For example, Claude-3-5-sonnet from June 2024 has difficulty following all instructions regarding
the return formats.
Claude-3-5-sonnet from October 2024 follows these instructions much better.

## Add Response Models

You can find the full source code of this 4th approach here: [4-response-model.py](./4-response-model.py).

As a last step I looked at the `Response Models` offered by Agno.
They make sure that the data returned fits a pre-defined structure and can
be treated afterwards using non-agentic programming.
For this to work, I created two classes to describe the formats required:

```python
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
```

The `RSEDescription` holds all the fields I'm interested in, and the `description=` is passed by Agno
to the LLM.
This means that, for once, comments are actually useful!
I added a `csv` method to the class to being able to return the values as one line of CSV.
`RSEs` then holds a list of `RSEDescription`, so that the model can return a list of RSEs it found.
Adding these models to the Agent is again very simple:

```python
agent = Agent(
  [...]
  response_model=RSEs,
)
```

When calling the model, it is good to check that the output is actually of type `RSEs`, as the
LLM might return something not parseable by Agno:

```python
    run: RunResponse = agent.run(email)
    if isinstance(run.content, RSEs):
        for rse in run.content.rses:
            print(rse.csv())
```

### Instruction Adjustment

As always, it is good to only check the results on a couple of addresses, to have a fast test
cycle.
There were some edge cases that I had to adjust:

```text
  A Doctoral Assistant is NOT an RSE, so don't include them.
  An exception is if the Doctoral Assistant is the email given to you in the input.
  In this case, also include the Doctoral Assistant in the output.
```

And some of the models added text to the final output, which resulted in Agno not being able
to parse the output as JSON, so I also added:

```text
  In your final response, do not reflect on your thought process, but only
  return the final answer as JSON, as described.
```

Which reduced the wrong answers from Claude-3-5-sonnet by a lot.

## More Things to try out with Agno

This is but a simple agentic setup using Agno: one agent, two tools, and two response models.
But Agno can do a lot more, which I did not explore:

- [Teams](https://docs.agno.com/teams/introduction) to make different specialized LLMs work together nicely
- [Reasoning](https://docs.agno.com/reasoning/introduction) to work out how to best approach a task
- [Knowledge](https://docs.agno.com/knowledge/introduction) uses a vector database to store additional data
- [Storage](https://docs.agno.com/storage/introduction) for keeping information between different runs of the agent
- [Workflows](https://docs.agno.com/workflows/introduction) to implement complex interactions between all of this

## Look Ma - No LLMs

To finish, you can look at the non-agentic solution here: [5-normal.py](./5-normal.py).
It is even a bit shorter than the agentic instructions, and has some advantages:

- it is reproducible and deterministic (given the same internet-conditions as an agentic LLM)
- duplicates and edge cases are explicitely handled
- I can run it with debugging outputs to know _exactly_ what is happening
- it only consumes a small percentage of power compared to interacting with an LLM
- there is even a cache of the requests to EPFL's API, so during tests I don't spam EPFL...

Of course an LLM helped me writing the code, though, and I learnt things like
[@cache](https://docs.python.org/3/library/functools.html#functools.cache) and how to make this
persistent on disk.
But some of the problems were still faster with a Google search.

# Running the Code

You should use the excellent [devbox](https://www.jetify.com/devbox) to run this code.
If you don't have it yet installed, run the following command in your shell, or
follow the [installation instructions](https://www.jetify.com/docs/devbox/installing_devbox/):

```bash
# Install Devbox
curl -fsSL https://get.jetify.com/devbox | bash
```

Before you can run the code, you need to give access to one of the following models:

- Anthropic, by running `export ANTHROPIC_API_KEY=sk-ant-...`
- OpenAI, by running `export OPENAI_API_KEY=sk...`
- LM Studio server running on the port 1234

If you're using another model, please have a look at [Models](https://docs.agno.com/models/introduction)
to see if it is supported, and how you need to configure it.
Once all is set up, you can run any of the examples with this command:

```bash
devbox run 1-naive
```

Of course you can replace `1-naive` with `2-curl`, `3-epfl-api`, `4-response-model`, or `5-normal`.
If you want to play around with it, you can also start a shell with the python environment in it:

```bash
devbox shell
```

And then use `python`, `pip`, as normal.
Everything will be kept to your directory.

# Agents or not Agents, that is the Question

This has been a simple example how to use agents for a problem where a non-agentic
approach is also possible.
As such I was not impressed by the agents - the edge cases where much more numerous than
what I tought beforehand.
Debugging agents is also more difficult and slower than debugging non-agentic code:

- How do you repeatedly create the same requests to an LLM?
- How to write unit-tests or general test-cases for these agents?
- Why does it all of a sudden fail? Oh wait, restarting makes it work...

But I already have new ideas what I want to try: I'm very disappointed by [Cursor](https://cursor.com)
to program something it doesn't know well.
Could a good agentic team do better programming?
How would I have to interface this with VisualStudio / Zed / Vim?

Stay tuned for more agents, this is fun :)

Linus Gasser, for https://C4DT.epfl.ch