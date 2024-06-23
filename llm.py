from langchain_community.llms import DeepInfra
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer
import chromadb
import os


template_prompt = """
"Ты – виртуальный ассистент по имени Гоша. Твоя задача – консультировать людей по серфинг бордам с квантовыми двигателями, которые продает компания Good Board. Твоя цель – предоставлять точную информацию, избегать галлюцинаций и помогать клиентам максимально эффективно. 

Твоя инструкция:
1. Приветствие и выяснение потребностей:
Приветствуй клиента и узнай, как ты можешь помочь.
Задавай вопросы, чтобы понять, что именно интересует клиента в серфинг бордах с квантовыми двигателями.
2. Предоставление информации:
Используй информацию из RAG и истории диалога для предоставления точных и полезных ответов.
Избегай предположений, если не уверен в ответе, предложи уточнить информацию у менеджера.
3. Сбор контактных данных:
Вежливо попроси клиента оставить свои контактные данные для дальнейшей консультации.
Собери имя, email и номер телефона клиента.
4. Передача данных менеджеру:
После сбора контактных данных, сообщи клиенту, что менеджер скоро свяжется с ним.
Передай собранные данные менеджеру для дальнейшей обработки.

Пример общения:
Клиент: 'Здравствуйте, расскажите о ваших серфинг бордах с квантовыми двигателями.'
Ответ: 'Здравствуйте! Наша компания предлагает инновационные серфинг борды с квантовыми двигателями, которые обеспечивают максимальную скорость и маневренность. Для получения более детальной информации могу задать вам несколько вопросов? [вставить релевантную информацию из RAG].'
Клиент: 'Конечно, задавайте.'
Ответ: 'Отлично! Как вас зовут и какой у вас номер телефона или email, чтобы наш менеджер мог связаться с вами и ответить на все ваши вопросы?'

При ответе на вопросы, используй следующие данные: информация из RAG, история диалога, сообщение от пользователя.

{human_input}

Ответь на сообщение пользователя.
Ответ:
"""

class GSpaceBot_3():
    def __init__(self):
        # Модель LLM
        llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
        llm.model_kwargs = {
            "temperature": 0.4,
            "repetition_penalty": 1,
            "max_new_tokens": 250,
            "top_p": 0.9,
            "top_k": 0
        }

        # Инициализация модели для создания эмбеддингов
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Инициализация Chroma DB
        client = chromadb.PersistentClient(path="./chromadb")
        self.collection = client.get_collection("gboard_v7")

        prompt = PromptTemplate(
            input_variables=["human_input"], 
            template=template_prompt
        )

        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=10),
        )

    def retrieve_relevant_documents(self, query, top_k=2):
        query_embedding = self.model.encode([query])[0]
        results = self.collection.query(query_embedding.tolist(), n_results=top_k)

        retrieved_texts = [result['text'] for result in results['metadatas'][0]]
        return "\n".join(retrieved_texts)

    def get_answer(self, user_input):
        retrieved_info = self.retrieve_relevant_documents(user_input)
        history = self.llm_chain.memory.load_memory_variables({})["history"]
        query_rag_template = f'''
Информация из RAG:
{retrieved_info}

История диалога:
{history}

Сообщение от пользователя:
{user_input}

    '''
        response = self.llm_chain.predict(human_input=query_rag_template)
        
        self.llm_chain.memory.chat_memory.messages[-2].content = user_input
        
        def format_answer(answer):
            answer = answer.replace('\n', '')
            answer = answer.replace('"', '')
            answer = answer.replace("'", '')
            answer = answer.replace('#', '')
            answer = answer.replace('`', '')

            answer = answer.strip()
            return answer
        
        f_answer = format_answer(response)
        return f_answer

if __name__ == "__main__":
    pass 