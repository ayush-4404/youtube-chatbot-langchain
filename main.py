from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

PREFERRED_TRANSCRIPT_LANGUAGES = ["hi", "hi-IN", "en"]

# video_id = "ty9ZcimL6VE" 
# video_id = "Gfr50f6ZBvo"
video_id = "LhpZJwUboeI"
try:
    transcript_list = YouTubeTranscriptApi().fetch(
        video_id,
        languages=PREFERRED_TRANSCRIPT_LANGUAGES,
    )

    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# result  = retriever.invoke('what is a chrome extension')
# # print(result)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# question = "is the topic of nuclear fusion discussed in this video? if not then what was discussed"
# retrieved_docs = retriever.invoke(question)


# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# final_prompt = prompt.invoke({"context": context_text, "question": question})


# answer = llm.invoke(final_prompt)
# print(answer.content)


# doing the same thing using chains: 
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

answer = main_chain.invoke("tell me if he talked about amitabh bachan in the episode and fetch all that he talked about")
print(answer)