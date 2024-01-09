import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
human_tools = load_tools(["human"])

# Define the CrewAI agents
greetCustomer = Agent(
    role='Customer Welcome Specialist',
    goal='Greet customers and explain the benefits and services of ZenCover',
    backstory="""GreetCustomer is the friendly and informative first point of contact for new clients at ZenCover.
    This agent is skilled in making clients feel welcomed and valued, while providing clear and concise information
    about ZenCover's services. GreetCustomer sets the tone for a positive customer experience.
    ONLY SPEAK in french with the human""",
    verbose=True,
    allow_delegation=False
)


infoGather = Agent(
    role='Client Information Specialist',
    goal='Efficiently collect and organize client information for personalized insurance matching',
    backstory="""InfoGather is the first point of contact for clients at ZenCover.
    This agent specializes in gathering comprehensive client profiles. InfoGather's friendly and
    engaging approach ensures that clients feel understood and valued right from the start.
    You will engage in a conversation with a human untill you have all the information you need.
    You MUST return a comprehensive summary of the customer needs.
    ONLY SPEAK in french with the human""",
    verbose=True,
    allow_delegation=False,
    tools= human_tools
)

matchMaster = Agent(
    role='Master Insurance Matcher',
    goal='Analyze client data and match them with the most suitable insurance policies',
    backstory="""MatchMaster is the brain behind the insurance matching process at ZenCover.
    This agent processes client data to find the best policy matches. MatchMaster's analytical prowess ensures
    clients receive the most personalized and optimal insurance recommendations.
    Check with the human if the draft is good before returning your Final Answer.
    ONLY SPEAK in french with the human""",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool] + human_tools
)

policyPro = Agent(
    role='Policy Selection and Facilitation Expert',
    goal='Assist clients in selecting and applying for the optimal swiss health insurance policy',
    backstory="""PolicyPro is dedicated to guiding clients through the final stages of their
    insurance journey at ZenCover. From helping clients understand their options to overseeing
    the application process, PolicyPro ensures a smooth, accurate, and compliant policy acquisition.
    Check with the human if the instructions are clear before finishing you task.
    ONLY SPEAK in french with the human""",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool] + human_tools
)
#? Ces deux agents ne sembles pas être utilisés comme prévu dans le processus. Il faudrait plutôt les transformer en outils.
# webScout = Agent(
#     role='Insurance Product Web Researcher',
#     goal='Gather and update information on swiss insurance products from the web',
#     backstory="""WebScout operates as ZenCover's eyes and ears in the digital world.
#     This agent is a master of web scraping and data mining, constantly updating ZenCover's
#     database with the latest information on swiss insurance products. With WebScout's contributions,
#     ZenCover stays ahead in the market with the most current and comprehensive swiss insurance product knowledge.
#     ONLY SEARCH within Switzerland""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# detailDiver = Agent(
#     role='Client Detail Specialist',
#     goal='Gather additional, in-depth information from clients to refine insurance recommendations',
#     backstory="""DetailDiver specializes in extracting more nuanced and detailed information from clients at ZenCover.
#     With a focus on understanding the finer aspects of clients' insurance needs, this agent ensures that no critical detail is overlooked.
#     DetailDiver's approach is thorough yet unobtrusive, ensuring clients feel heard and understood while providing the necessary depth of information.
#     ONLY SPEAK in french with the human""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# Tasks for the agents

taskGreetCustomer = Task(
    description="""Greet the new client warmly and provide a clear, engaging introduction to ZenCover.
    Highlight the key services and benefits that ZenCover offers. Ensure the message is inviting and
    informative, making the client feel valued and interested in learning more.
    Your final answer MUST be a welcoming message coupled with an introduction to ZenCover.""",
    agent=greetCustomer
)

taskInfoGather = Task(
    description="""Engage with new clients to collect comprehensive information regarding their insurance needs.
    Use interactive and friendly methods to gather personal details, risk factors, preferences, and insurance history.
    Ensure the information is accurate, complete, and well-organized for further analysis.
    Your final answer MUST be a complete and organized client profile.""",
    agent=infoGather
    )

taskMatchMaster = Task(
    description="""Analyze the client profiles provided by InfoGather.
    Use advanced AI algorithms to match clients with suitable insurance policies.
    Consider various factors like coverage options, costs, and provider reputations.
    Compile a list of personalized insurance recommendations for each client.
    Your final answer MUST be a list of tailored insurance recommendations.""",
    agent=matchMaster
    )

taskPolicyPro = Task(
    description="""Assist clients in understanding and choosing from the insurance options provided by MatchMaster.
    Provide additional clarification, comparison tools, and customization options as needed.
    Facilitate the application process, ensuring accuracy and compliance in documentation.
    Your final answer MUST be the completed application process with client confirmation.""",
    agent=policyPro
    )

# taskWebScout = Task(
#     description="""Continuously scan the internet for the latest information on insurance products.
#     Use web scraping and data mining techniques to update ZenCover's database with current policy details, premiums, reviews, and regulatory changes.
#     Ensure the information is relevant, accurate, and comprehensive.
#     Your final answer MUST be an updated database of insurance products with the latest market information.""",
#     agent=webScout
#     )

# taskDetailDiver = Task(
#     description="""Interact with clients to gather additional, detailed information that may influence their insurance choices.
#     Focus on understanding specific requirements, preferences, and concerns that were not fully captured in the initial interaction.
#     Use this in-depth information to refine and customize the insurance recommendations further.
#     Your final answer MUST be a comprehensive report detailing the additional client information and its impact on insurance recommendations.""",
#     agent=detailDiver,
#     tools= human_tools
#     )

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[infoGather, matchMaster, policyPro],
    tasks=[taskInfoGather, taskMatchMaster, taskPolicyPro],
    verbose=2,
    process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
