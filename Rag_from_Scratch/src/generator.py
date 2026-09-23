from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
class Generator:
    def __init__(self,model):
        self.model=ChatGoogleGenerativeAI(
            model=model
        )
        
    def generate(self, question, context):
        
        prompt = f"""
        You are a question-answering assistant.

        Use only the provided context to answer the question.

        Rules:
        - If the answer is present in the context, answer clearly and directly.
        - If the answer is not present in the context, say "I don't know."
        - Do not use outside knowledge.
        - Do not make up information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        answer = self.model.invoke(prompt)
        return answer.content[0]["text"]