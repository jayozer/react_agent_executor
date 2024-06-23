from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"') # strip away non alphabetic chars
    return len(text)




if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    #print(get_text_length("Hello ReAct LangChain!"))
    tools = [get_text_length]

    template="""
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
# Last time used tools from langchain but also can use a tool decorator that takes a function and turn it into a langchain tool. 

    llm =ChatOpenAI(temperature=0, stop=["\nObservation"])
    agent = {"input": lambda x:x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of 'Hello ReAct LangChain!' in characters?'"})
    print(agent_step)
