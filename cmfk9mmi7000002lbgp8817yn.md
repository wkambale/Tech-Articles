---
title: "Build Your First AI Agent with Gemini and LlamaIndex"
seoTitle: "Build Your First AI Agent with Gemini and LlamaIndex"
seoDescription: "Building a Tourism AI Assistant with Agentic Workflows"
datePublished: Sun Sep 14 2025 22:27:20 GMT+0000 (Coordinated Universal Time)
cuid: cmfk9mmi7000002lbgp8817yn
slug: build-your-first-ai-agent-with-gemini-and-llamaindex
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1757888693269/ee93cfd2-3554-4dbf-baf2-4d7f2867425f.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1757888787651/f3e8974d-1a20-44b2-93b1-805982d9a788.png
tags: tourism, ai-agents, llamaindex, gemini

---

The world of LLMs is moving beyond simple chatbots. The new frontier is **AI agents**: systems that can reason, plan, and use external tools to accomplish complex tasks. In tourism, this means our assistant can automatically fetch up-to-date info on destinations, guide users to attractions or restaurants, find relevant images, and even draft promotional posts. Under the hood, we use Google’s advanced Gemini model as the “brain” and LlamaIndex to connect Gemini to custom Python tools. The agent follows the **ReAct (Reason+Act)** framework: it thinks about the query, chooses tools, acts (calls them), then reasons over the results. The result is a flexible travel assistant that can handle multi-step tasks in a human-like way.

Our agent will demonstrate how to:

* **Combine LLMs and Tools:** Overcome LLM limitations by letting them call Python functions (tools) for fresh data.
    
* **Build Custom Tools:** Write scrapers and generators (e.g. for attractions and images) that the agent can invoke.
    
* **Use Gemini as the LLM:** Leverage Google’s multimodal Gemini 2.5 Pro model (our most advanced AI model) for reasoning.
    
* **Leverage LlamaIndex:** Use LlamaIndex’s `ReActAgent` and `FunctionTool` wrappers to glue tools and the LLM together.
    
* **See the ReAct Loop in Action:** Peek at the agent’s chain of thought as it solves a query by sequentially “thinking” and “acting”.
    
* **Advanced Prompting:** Go beyond raw data by feeding tool outputs into a carefully crafted prompt for insightful recommendations.
    

By the end, you’ll have a working Tourism AI Assistant that answers travel questions, scrapes live data, finds images, and even tweets about destinations—all powered by Gemini and LlamaIndex.

## Our Technology Stack

Before coding, let’s understand the pieces we’ll use:

* **Google Gemini 2.5 Pro:** A state-of-the-art multimodal LLM (text+code+images+video) with strong reasoning capabilities. This is the “brain” of our agent. We use the `models/gemini-2.5-pro` endpoint via Google’s AI Generative SDK.
    
* **LlamaIndex:** An open-source framework for building LLM apps. It provides the `ReActAgent` class and `FunctionTool` wrapper to connect LLMs with Python tools. LlamaIndex lets us expose any function to the agent in a structured way.
    
* **Web Scraping Libraries:** We’ll use `requests` and `beautifulsoup4` to fetch live data from travel sites (for attractions, etc.). (In production, a tourism API is often preferred to avoid brittle scrapers, but for our tutorial we’ll show how to scrape responsibly.)
    
* **Image Search:** We’ll implement a simple DuckDuckGo image scraper to let the agent fetch pictures of destinations.
    
* **Content Generator:** A function to draft a promotional social-media post about a place or event, illustrating how the agent can take action, not just fetch data.
    

## Installing Dependencies

First, install the required Python packages. In your notebook or script, run:

```bash
!pip install -q llama-index llama-index-llms-gemini google-generativeai python-dotenv beautifulsoup4 requests
```

* **llama-index-llms-gemini:** Adds Gemini support to LlamaIndex.
    
* **google-generativeai:** Google’s SDK to call the Gemini API.
    
* **python-dotenv:** For loading API keys from a `.env` file (or Colab secrets).
    
* **beautifulsoup4 & requests:** For our web-scraping tools.
    

We’ll also enable detailed logging to trace the agent’s reasoning:

```python
import os, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## Authentication

To use Gemini, you need a Google AI Studio API key:

1. Create an API key in Google AI Studio.
    
2. Store it as `GOOGLE_API_KEY` in your environment (or Colab secrets).
    

Then configure the Google Generative AI SDK:

```python
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # if using .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
logger.info("Configured Google API key.")
```

## Configuring the LLM and Embedding Model

We tell LlamaIndex which LLM (Gemini) and embedding model to use. The embedding model isn’t crucial here (we won’t do retrieval-heavy tasks), but LlamaIndex requires one. We’ll use a small HuggingFace model:

```python
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODEL_NAME = "models/gemini-2.5-pro"  # Google’s Gemini 2.5 Pro model

logger.info("Configuring LlamaIndex settings...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Gemini(model=MODEL_NAME)  # uses GOOGLE_API_KEY automatically
logger.info(f"Using Gemini LLM: {MODEL_NAME}")
```

Note: Gemini 2.5 Pro is a powerful model that excels at complex tasks. It supports text, code, and images, and even has built-in “thinking” ability for chain-of-thought reasoning.

## Crafting Our Tools

An agent is only as capable as its tools. A **tool** here is a Python function that performs some action—like fetching data or creating content—that the LLM alone cannot do (e.g. real-time web queries). We’ll create a small toolkit for tourism tasks:

* **Tool 1:** `get_city_attractions(city)` – Scrape top attractions for a city. (As an example, we’ll scrape a known travel guide or Wikipedia for a few highlights.)
    
* **Tool 2:** `get_city_restaurants(city)` – (Optional) Scrape or list popular restaurants. For brevity, this could return a static list or use a simple scrape.
    
* **Tool 3:** `search_for_destination_images(place)` – Search DuckDuckGo for images of a place (like “Eiffel Tower Paris”). Returns a few image URLs.
    
* **Tool 4:** `generate_tourism_post(place, summary)` – Generate a friendly promotional social-media post (tweet) about an attraction or city, given its name and a short summary.
    

Each tool will have a clear docstring explaining its purpose and inputs. The agent reads these docstrings to know when to use which tool. For example:

```python
import requests
from bs4 import BeautifulSoup

def get_city_attractions(city: str) -> str:
    """
    Fetches the top tourist attractions for the given city.
    Returns a bullet list of attractions and brief info.
    """
    try:
        # Example: scrape PlanetWare or Wikipedia page for top attractions
        url = f"https://www.planetware.com/{city.lower()}/top-rated-tourist-attractions-in-{city.lower()}.htm"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        attractions = [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:5]]
        if not attractions:
            raise Exception("No attractions found")
        return "\n".join(f"- {a}" for a in attractions)
    except Exception as e:
        return f"Could not retrieve attractions for {city} (error: {e})"
```

```python
def get_city_restaurants(city: str) -> str:
    """
    Returns a short list of popular restaurants in the city.
    (For demo purposes, this may be a static list or scraped from a simple source.)
    """
    # Placeholder: in a real app, use an API or reliable source
    dummy_data = {
        "Paris": ["Le Jules Verne (Eiffel Tower)", "L'Ambroisie", "Septime"],
        "Rome": ["Da Enzo al 29", "Roscioli", "La Pergola"]
    }
    return "\n".join(f"- {r}" for r in dummy_data.get(city, ["No data available"]))
```

```python
def search_for_destination_images(query: str) -> list:
    """
    Searches DuckDuckGo Images for the query and returns a list of image URLs.
    """
    try:
        res = requests.get(f"https://duckduckgo.com/?q={query.replace(' ', '+')}&iar=images&iax=images")
        soup = BeautifulSoup(res.text, "html.parser")
        imgs = soup.select("img.tile--img__img")[:5]
        return [img.get("src") for img in imgs if img.get("src")]
    except Exception as e:
        return [f"Image search failed: {e}"]
```

```python
def generate_tourism_post(place: str, summary: str) -> str:
    """
    Generates a friendly social media post about a place.
    """
    prompt = f"Write an enthusiastic tweet about visiting {place}. Summary: {summary}"
    resp = genai.generate_message(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return resp.last
```

*Note:* In production, scraping arbitrary sites (like PlanetWare) can be fragile. Sites may block scrapers, change layout, or have legal terms against scraping. For robust applications, a dedicated travel data API is preferable. Here we use simple scrapers for illustration.

## Wrapping Functions for LlamaIndex

LlamaIndex needs tools wrapped in `FunctionTool` objects. This exposes each function’s signature and docstring to the LLM. Docstrings become part of the agent’s understanding of when to use each tool. For example:

```python
from llama_index.core.tools import FunctionTool

logger.info("Wrapping functions into LlamaIndex tools...")
tools = [
    FunctionTool.from_defaults(fn=get_city_attractions,
                               name="get_city_attractions",
                               description="Get the top tourist attractions in a city."),
    FunctionTool.from_defaults(fn=get_city_restaurants,
                               name="get_city_restaurants",
                               description="Get popular restaurants in a city."),
    FunctionTool.from_defaults(fn=search_for_destination_images,
                               name="search_for_destination_images",
                               description="Search for images of a place."),
    FunctionTool.from_defaults(fn=generate_tourism_post,
                               name="generate_tourism_post",
                               description="Create a social media post promoting an attraction or city.")
]
logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")
```

Each `FunctionTool` includes the function name, parameters, and a human-readable docstring. The agent will **choose which tool to call** by looking at your query and the tool descriptions, then pass the right arguments.

## Initializing the ReAct Agent

With tools ready, we create the agent. We use LlamaIndex’s `ReActAgent`, which implements the Reason-and-Act loop. The agent will **think** about the user’s question, decide **which tool** to use and with what inputs, **execute** the tool, then observe the result and repeat as needed. Setting `verbose=True` lets us see the entire chain of thought:

```python
from llama_index.core.agent import ReActAgent

logger.info("Initializing the ReAct agent...")
agent = ReActAgent.from_tools(tools=tools, llm=Settings.llm, verbose=True)
logger.info("Tourism Agent is ready to go!")
```

Under the hood, `ReActAgent` will prompt Gemini with a system message describing the tools and this reasoning framework. The LLM will follow a structure like:

```bash
Thought: I should use [tool] because ...
Action: [tool] with input {...}
Observation: (tool output)
Thought: ... continue or final answer.
```

This explicit reasoning flow is characteristic of the ReAct approach.

## Trying Out the Tourism Agent

Our agent is now online! Let’s test it with a couple of example queries and inspect its reasoning (verbose output):

### Scenario 1: Simple Question

**User:** “What are the top attractions in Paris?”

```plaintext
response = agent.chat("What are the top tourist attractions in Paris?")
print(response)
```

*Agent’s (verbose) Thought Process:*

* **Thought:** The user asks for top attractions in Paris. The `get_city_attractions` tool seems appropriate.
    
* **Action:** get\_city\_attractions
    
* **Action Input:** `{"city": "Paris"}`
    
* **Observation:**
    
    ```bash
    - Eiffel Tower
    - Louvre Museum
    - Notre-Dame Cathedral
    - Sacré-Cœur Basilica
    - Musée d'Orsay
    ```
    
* **Thought:** I have the list of attractions. Now I answer the user using this info.
    
* **Answer:** *“Top attractions in Paris include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Sacré-Cœur, and Musée d’Orsay.”*
    

**Agent Answer:** *Top attractions in Paris include: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Sacré-Cœur Basilica, and Musée d’Orsay.*

This shows the ReAct loop: the agent identified the need for a tool, executed it, and synthesized the output.

### Scenario 2: Multi-Step Task

**User:** “Find art-related attractions in Paris and then write a tweet about them.”

This requires multiple steps: identify art attractions, then generate a tweet.

```python
response = agent.chat("Find art-related attractions in Paris and draft a tweet about them.")
print(response)
```

*Agent’s Thought Process:*

* **Thought:** The user needs two things: 1) art-related attractions in Paris, 2) a tweet about them. First, get all attractions.
    
* **Action:** get\_city\_attractions
    
* **Input:** `{"city": "Paris"}`
    
* **Observation:** (same list as before)
    
* **Thought:** The attractions list is obtained. Now pick those related to art (e.g. museums, galleries). These include Louvre Museum and Musée d'Orsay. I will create a summary and use the `generate_tourism_post` tool.
    
* **Action:** generate\_tourism\_post
    
* **Input:** `{"place": "Paris Museums", "summary": "Highlights: Louvre Museum and Musée d'Orsay"}`
    
* **Observation:**
    
    ```bash
    Explore Paris's Artistic Treasures!
    Don’t miss the Louvre Museum’s timeless classics and the Musée d’Orsay’s Impressionist masterpieces. From ancient wonders to modern art, Paris has it all. Bon voyage and happy exploring!
    ```
    
* **Thought:** I have the tweet text. Now I can share the final answer.
    
* **Answer:** *Here’s a tweet promoting Paris’s art attractions:… (tweet above)…*
    

**Agent Answer:** *“Here’s a tweet about Paris’s art attractions: Explore Paris’s Artistic Treasures! Don’t miss the Louvre Museum’s timeless classics and the Musée d’Orsay’s Impressionist masterpieces. From ancient wonders to modern art, Paris has it all. Bon voyage and happy exploring!”*

This demonstrates chaining: the agent used `get_city_attractions`, filtered the results, then used `generate_tourism_post` with the summary. The LLM reasoned at each step, as expected in a ReAct agent.

## Advanced Prompting: Personalized Recommendations

So far we’ve fetched raw data and posted about it. But LLMs can also provide **insights** on top of data. For example, suppose a user asks:

*“I’m a history buff visiting Rome. Which attractions should I see?”*

Simply returning a list of *all* Rome attractions isn’t ideal. Instead, we want **personalized recommendations**. We can achieve this by combining our tools with a clever prompt:

1. **Get raw data via tool:** Call `get_city_attractions("Rome")`.
    
2. **Engineer a focused prompt:** Tell Gemini to read the list and pick the best 3-5 for a history enthusiast, explaining why.
    
3. **Generate the answer with Gemini:** The LLM acts as a reasoning layer on the data.
    

For example:

```python
# Step 1: fetch attractions
attractions = get_city_attractions("Rome")
# (assume this returns a bullet list of sites like Colosseum, Vatican, Pantheon, etc.)

# Step 2: craft a targeted prompt
prompt = f"""
You are an expert tour guide for Rome. A user is interested in historical sites. 
From the following list of Rome attractions, recommend 3-5 must-see locations for a history buff. Explain why each is a good choice.
Rome Attractions:
{attractions}

Focus on historical significance and include brief descriptions.
"""
# Step 3: call Gemini directly with this prompt
historical_resp = genai.generate_message(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
print(historical_resp.last)
```

**Sample Output:**

> Excellent choices for a history enthusiast in Rome include:
> 
> * **Colosseum** – The iconic ancient amphitheater where gladiators once fought. A symbol of Rome’s imperial past and architecture.
>     
> * **Roman Forum** – The heart of ancient Rome’s political and social life. You can walk among ruins of temples and government buildings.
>     
> * **Pantheon** – A 2,000-year-old temple-turned-church, showcasing Rome’s engineering and dedication to the gods. Its monumental dome is a marvel.
>     
> * **Vatican Museums & St. Peter’s Basilica** – While a church, the Vatican holds vast historical and artistic treasures spanning millennia, including Raphael’s Rooms and Michelangelo’s Pietà.
>     
> * **Catacombs of Callixtus** – Underground burial chambers that reveal early Christian history and traditions.
>     
> 
> Each of these sites offers rich historical insights into Rome’s past civilizations and will captivate any history buff!

This “reasoning layer” approach leverages Gemini’s understanding to *analyze and filter* the tool output. We transformed a raw list into a personalized recommendation list with explanation. One could even wrap this workflow into a new tool (e.g. `recommend_attractions_for_interest`) for the agent to use directly.

## Conclusion

We’ve built a fully functional **Tourism AI Assistant** that can interpret user requests, choose the right tool, fetch live data, and present it in a helpful way. We saw:

* **Setting up the agent:** Configuring Gemini and LlamaIndex.
    
* **Writing tools:** Python functions with clear docstrings, wrapped as `FunctionTool` for the agent.
    
* **Using ReAct:** The agent’s chain of thought (“Thought”, “Action”, “Observation”) shows how it plans and executes.
    
* **Multimodal capability:** Although we didn’t show it here, Gemini supports images and code, so you could extend this agent to analyze photos of landmarks or compute routes.
    
* **Advanced prompting:** We enhanced raw data by feeding it into Gemini with a crafted prompt, yielding richer, customized advice.
    

This agentic architecture is highly extensible. Next steps might include integrating real travel APIs (for flights or hotels), adding a **memory** so the assistant recalls user preferences, or connecting to mapping services for directions. The key idea is that the LLM does the reasoning, while we supply specialized tools for any real-world data or action.

*Happy travels and happy building!*