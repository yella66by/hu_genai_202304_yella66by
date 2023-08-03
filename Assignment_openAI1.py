
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

#content="John wants to make a pizza. First, he needs to buy the ingredients. Once he has all the ingredients, he will start preparing the pizza dough. After that, he will add the sauce and toppings and bake the pizza in the oven. Finally, he can enjoy the delicious homemade pizza."

content ="For christmas shopping, first we need to have some money. Then we can go shopping, and then think what we want to buy. The things we can buy are iPhone, car or laptop."

llm =OpenAI(
        temperature=0,
        openai_api_key='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',
        model_name='text-davinci-003',
        top_p=1
)

def count( chain,query,doing):
    with get_openai_callback() as cb:
        result=chain.run(query)
        print(f'spent a total of {cb.total_tokens} tokens {doing}')
    return result
conversation_buf =ConversationChain(
     llm=llm,
    memory=ConversationBufferMemory()
)
prompt = f"""
    Your task is to break down the content into a series of steps in English.
    write those steps in the following format:
    Step1
    Step2
    .
    .
    .
    StepN


    <content>:{content}

    """
response = count(conversation_buf, prompt,"TO convert to steps")
print(response)
prompt2 = f"""
    Your task is to write the mermaid code from the previously generated steps and explanation for how to generate the code is delimited by triple quotes

    ```<steps>:
        Step 1: Buy the ingredients needed to make pizza. 
        Step 2: Prepare the pizza dough. 
        Step 3: Add the sauce and toppings to the pizza dough. 
        Step 4: Bake the pizza in the oven. 
        Step 5: Enjoy the delicious homemade pizza.

    <code>: graph TD
            A(Start) --> B(Buy Ingredients)
            B --> C(Prepare Burger Dough)
            C --> D(Add Sauce & Toppings)
            D --> E(Bake in Oven)
            E --> F(Enjoy Homemade Burger)

    Explanation of mermaid code:
    1) graph LR: This line indicates that we are creating a flowchart from left to right (LR).
    2)Nodes are represented by letters in parentheses, e.g., A(Start) represents the starting point of the process, and F(Enjoy Pizza) represents the endpoint.
    3)Arrows (-->) indicate the flow of the process from one node to another.
    Here's the step-by-step logic behind converting the text into Mermaid code:
    1)Identify the sequence of actions: In the text, we see a sequence of actions: "buy ingredients," "prepare dough," "add sauce & toppings," "bake in oven," and "enjoy pizza."
    2)Assign nodes: Assign a unique node (letter or word in parentheses) to each action. Here, we use letters A to F to represent the actions.
    3)Define the flow: Connect the nodes with arrows to represent the flow of actions in the process.
    This is a simple example, but in more complex scenarios, you can use additional Mermaid features, such as conditional statements (if-else), loops, or different types of diagrams 
    (e.g., sequence diagrams, Gantt charts) to represent different types of information based on the text description. Mermaid's flexibility allows you to create various visualizations 
    based on the data provided in the text.```

    <steps>:
    {response}

    <code>:
    Generate code only
    """
code=count(conversation_buf, prompt2,"To Generate code")
print(code)
