from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms.openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def qa_tool_chain(query):
    result = qa_chain({"question": query})
    return result["answer"]


text_loader = TextLoader('/home/sv/Downloads/Nexus-Downloads/BackUp-Downloads/ramayan.txt')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# documents = text_loader.load()

documents = text_loader.load_and_split(text_splitter=text_splitter)

embedding_func = OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents=documents, embedding=embedding_func)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=ChatOpenAI(model='gpt-3.5-turbo'),
    retriever=retriever
)

qa_tool = Tool(
    name="Retriever QA",
    func=qa_tool_chain,
    description="Answer questions based on retrieved documents"
)


tools = [qa_tool]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model='gpt-3.5-turbo'),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query = "Who is the greatest of all time between Lord Rama and Ravana?"

response = agent.run(query)

print(response)

