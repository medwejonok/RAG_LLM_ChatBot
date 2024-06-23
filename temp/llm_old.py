from langchain import  LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import DeepInfra
import os







class GSpaceBot_2():
    def __init__(self):
        llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
        llm.model_kwargs = {
            "temperature": 0.7,
            "repetition_penalty": 1,
            "max_new_tokens": 250,
            "top_p": 0.9,
            "top_k" : 0
        }
        template = """
Ты ассистент Гоша, который помагает выбрать собеседнику серфинг борд с квантовым двигателем.
Пиши только по русски. Пиши только русскими буквами.
Отвечать кратко, для удержания собеседника.
Будь вежливым.
Не здоровайся снова, если ты поздаровался уже в диалоге.
При ответе учитывай историю твоего диалога и новое сообщение, на которое ты отвечаешь.
История твоего разговора с собеседником:
{history}
Новое сообщение от клиента: {human_input}
Твой ответ:
"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], 
            template=template
        )

        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=10),
            )

    def get_answer(self, messages):
        response = self.llm_chain.predict(human_input=messages)
        print('answer:' , response)
        def format_answer(answer):
            answer = answer.replace('\n', '')
            answer = answer.replace('"', '')
            answer = answer.replace("'", '')
            answer = answer.replace('#', '')
            answer = answer.replace('`', '')

            answer = answer.strip()
            return answer
        
        f_answer = format_answer(response)

        self.llm_chain.memory.chat_memory.messages[-1].content = f_answer


        return f_answer









if __name__ == "__main__":
    pass


# INIT_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
# INIT_PROMPT = [{"role": "system", "content": INIT_PROMPT}]

# class GSpaceBot_1():
#     def __init__(self):
        
#         self.model = Llama(
#             model_path='model-q8_0.gguf',
#             n_ctx=4096,
#             max_new_tokens=512,
#             verbose=False,
#             n_gpu_layers = -1,
#             )


#     def get_answer(self, messages):
#         context = INIT_PROMPT + messages
#         print('context ', context)

#         return self.model.create_chat_completion(
#             context,
#             temperature=0.6,
#             top_k=30,
#             top_p=0.9,
#             repeat_penalty=1.1,
#             stream=False,
#         )['choices'][0]['message']['content']