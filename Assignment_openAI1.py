import logging
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#content="John wants to make a pizza. First, he needs to buy the ingredients. Once he has all the ingredients, he will start preparing the pizza dough. After that, he will add the sauce and toppings and bake the pizza in the oven. Finally, he can enjoy the delicious homemade pizza."

content = "For christmas shopping, first we need to have some money. Then we can go shopping, and then think what we want to buy. The things we can buy are iPhone, car or laptop."

# Initializing the OpenAI instance
llm = OpenAI(
    temperature=0,
    openai_api_key='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',
    model_name='text-davinci-003',
    top_p=1
)

# Utility function to count tokens and run the language model chain
def count(chain, query, doing):
    with get_openai_callback() as cb:
        result = chain.run(query)
        logging.debug(f'Spent a total of {cb.total_tokens} tokens {doing}')
    return result

# Initializing the conversation buffer
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# Generating steps from the content
prompt = f"""
    Your task is to break down the content into a series of steps in English.
    Write those steps in the following format:
    Step1
    Step2
    Step3
    .
    .
    StepN-1
    StepN

    <content>:{content}

    """
response = count(conversation_buf, prompt, "TO convert to steps")
logging.info(f'Steps generated: {response}')

# Generating Mermaid code
prompt2 = f"""
    Your task is to generate the mermaid code from the previously generated steps and explanation for how to generate the code is delimited by triple quotes

    ```<steps>:
        {response}

    <code>: graph TD
    A[Christmas] -->|Get money| B(Go shopping)
    B --> C(Let me think)
    C -->|One| D[Laptop]
    C -->|Two| E[iPhone]
    C -->|Four| F[fa:fa-car Car]


    Explanation of mermaid code:
    1) graph LR: This line indicates that we are creating a flowchart from left to right (LR).
    2) Nodes are represented by letters in parentheses, e.g., A(Start) represents the starting point of the process, and F(Enjoy Pizza) represents the endpoint.
    3) Arrows (-->) indicate the flow of the process from one node to another.
    Here's the step-by-step logic behind converting the text into Mermaid code:
    1) Identify the sequence of actions: In the text, we see a sequence of actions: "buy ingredients," "prepare dough," "add sauce & toppings," "bake in oven," and "enjoy pizza."
    2) Assign nodes: Assign a unique node (letter or word in parentheses) to each action. Here, we use letters A to F to represent the actions.
    3) Define the flow: Connect the nodes with arrows to represent the flow of actions in the process.
    This is a simple example, but in more complex scenarios, you can use additional Mermaid features, such as conditional statements (if-else), loops, or different types of diagrams 
    (e.g., sequence diagrams, Gantt charts) to represent different types of information based on the text description. Mermaid's flexibility allows you to create various visualizations 
    based on the data provided in the text.```

    <steps>:
    {response}

    <code>:
    Generate code only
    """
code = count(conversation_buf, prompt2, "To Generate code")
logging.info(f'Generated Mermaid code: {code}')
print(code)