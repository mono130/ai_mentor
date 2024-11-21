import numpy as np
from sklearn.preprocessing import normalize
import Agent
import config
import setting
import openai
import DatabaseConnection
import time
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class AllAgent(Agent.Agent):

    def __init__(self):
        super().__init__()
        # 데이터베이스 설정 및 LLM 준비
        self.db = SQLDatabase.from_uri(f"mysql+pymysql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=config.api_key)
        self.context = self.db.get_context()
        self.table_info = self.context["table_info"]
        self.memory = ConversationBufferMemory(return_messages=True)  # 대화 기록 관리

    def set_examples_and_prompt(self, continuous):
        # continuous 값에 따라 다른 examples 및 few_shot_prompt 설정
        if continuous == 1:
            # 예시 데이터 설정 (연속 대화)
            self.examples = [
                {
                    "question": "해당 과목을 가르치는 교수님이 누구 있는지 말해줘",
                    "chat_memory": '[{"question":"기계학습은 무슨 학과에서 가르쳐?", "response":"기계학습은 컴퓨터인공지능학부에서 가르쳐."}]',
                    "sql_query": "SELECT 교과목명, 담당교수명, 학년, 학기, 수업목표 FROM lectures_info WHERE 학과 IN ('컴퓨터인공지능학부') AND 교과목명 = '기계학습';"
                },
                {
                    "question": "그럼 그 교수님의 수업은 뭐가 또 있지?",
                    "chat_memory": '[{"question":"CAD응용및실습은 어느 교수님이 가르쳐?", "response":"CAD응용및실습 (담당교수: 육기철)"}]',
                    "sql_query": "SELECT 교과목명, 학과, 학년, 학기, 수업목표 FROM lectures_info WHERE 담당교수명 = '육기철';"
                },
                {
                    "question": "그럼 그 학과의 커리큘럼 추천해줘.",
                    "chat_memory": '[{"question":"기계학습은 어느 학과에서 가르쳐?", "response":"기계학습은 통계학과에서 가르칩니다"}]',
                    "sql_query": "SELECT 학년 , 학기, 교과목명, 수업목표, 권장선수과목  FROM info_curriculum WHERE 학과 = '통계학과' AND 학년 IN (2, 3, 4);"
                },
                {
                    "question": "너가 말한 학과에서는 무슨 과목을 가르치는거야?",
                    "chat_memory": '[{"question":"정유진 교수님은 학과가 어디 소속이야?", "response":"정유진 교수님은 아동학과에 있습니다."}]',
                    "sql_query": "SELECT 교과목명, 담당교수명, 학년, 학기, 수업목표, 학과 FROM lectures_info WHERE 학과 = '아동학과';"
                },
                {
                    "question": "그 학과에 교수님은 누구있어?",
                    "chat_memory": '[{"question":"회계학과 3학년 1학기에 들을 수 있는 과목이 뭐가 있나요?", "response":"회계학과 3학년 1학기에 들을 수 있는 과목은 다음과 같습니다:...."}]',
                    "sql_query": "SELECT 교과목명, 담당교수명, 학년, 학기, 수업목표, 학과 FROM lectures_info WHERE 학과 = '회계학과';"
                },
                {
                    "question": "2학년이 들을 수 있는 과목은 뭐가있지?",
                    "chat_memory": '[{"question":"통계학과 3학년에 들을 수 있는 전공과목을 추천해줄 수 있어?", "response":"통계학과 3학년에 개설된 전공 과목은 다음과 같습니다..."}]',
                    "sql_query": "SELECT 교과목명, 담당교수명, 학년, 학기, 수업목표, 학과 FROM lectures_info WHERE 학년 = 2 AND 학과 = '통계학과';"
                },
                {
                    "question": "농업 빅데이터 전문가가 되려면 어떤 강의를 들어야 하나요?. 관련 학과:  스마트팜학과, 소프트웨어공학과, 컴퓨터인공지능학부, 생물산업기계공학과",
                    "chat_memory": '[{"question":"생물산업기계공학과 4학년에 들을 수 있는 전공과목을 추천해줄 수 있어?", "response":"생물산업기계공학과 4학년에 개설된 전공 과목은 다음과 같습니다..."}]',
                    "sql_query": "SELECT 교과목명, 수업목표, 학년, 학기, 학과, 권장선수과목 FROM lectures_info WHERE 학과 IN ('스마트팜', '소프트웨어공학과', '컴퓨터인공지능학부', '생물산업기계공학과') AND 학년 IN (2, 3, 4);"
                },
                {
                    "question": "위에 교수님들 모두 어디학과야? 관련학과: ,",
                    "chat_memory": '[{"question":"컴퓨터네트워크 과목을 가르치는 교수님은 누가있어?","response":"컴퓨터네트워크 과목을 가르치는 교수님은 다음과 같습니다: 김순영, 김민철, 김찬기, 편기현, 이종득."}]',
                    "sql_query": "SELECT 교과목명, 담당교수명, 학년, 학기, 수업목표, 학과 FROM lectures_info WHERE 교과목명 = '컴퓨터네트워크' AND 담당교수명 IN ('김순영', '김민철', '김찬기', '편기현', '이종득');"
                }
            ]
            
        else:
            # 예시 데이터 설정 (새로운 대화)
            self.examples = [
                {
                    "question": "김성찬 교수님은 뭘 가르쳐?",
                    "sql_query": "SELECT 교과목명, 학과, 학년, 담당교수명, 학기, 수업목표 FROM lectures_info WHERE 담당교수명 = '김성찬';"
                },
                {
                    "question": "소프트웨어공학과 3학년 1학기 과목 추천해줘",
                    "sql_query": "SELECT 교과목명, 수업목표, 담당교수명, 학년, 학기 FROM lectures_info WHERE 학과 = '소프트웨어공학과' AND 학년 = 3 AND 학기 = '1학기';"
                },
                {
                    "question": "통계학과의 커리큘럼을 추천해주세요.",
                    "sql_query": "SELECT 학년 , 학기, 교과목명, 수업목표, 권장선수과목  FROM info_curriculum WHERE 학과 = '통계학과' AND 학년 IN (2, 3, 4);"
                },
                {
                    "question": "차연수 교수님은 어느 학과에 계셔?",
                    "sql_query": "SELECT 학년 , 학기, 학과, 교과목명, 수업목표  FROM lectures_info WHERE 담당교수명 = '차연수';"
                },
                {
                    "question": "경제학과 3학년 1학기에 들을 수 있는 과목이 뭐가 있나요? 관련 학과:  경제부, 농경제유통학부, 회계학과, 무역학과, 경영학과, 공공인재학부, 일반사회교육과, 수학과, 지역산업학과, 농업시스템학과,",
                    "sql_query": "SELECT 학년, 학기, 교과목명, 수업목표, 담당교수명 FROM lectures_info WHERE 학과 IN ('경제부', '농경제유통학부', '회계학과', '무역학과', '경영학과', '공공인재학부', '일반사회교육과', '수학과', '지역산업학과', '농업시스템학과') AND 학년 = 3 AND 학기 = '1학기';"
                },
                {
                    "question": "AI기계결함 전문가가 되려면 어떤 강의를 들어야 하나요?. 관련 학과:  기계설계학과, 소프트웨어공학과, 항공우주공학과",
                    "sql_query": "SELECT 교과목명, 수업목표, 학년, 학기, 학과, 권장선수과목 FROM info_curriculum WHERE 학과 IN ('기계설계학과', '소프트웨어공학과', '항공우주공학과') AND 학년 IN (2, 3, 4);"
                },
                {
                    "question": "농업 빅데이터 전문가가 되려면 어떤 강의를 들어야 하나요?. 관련 학과:  스마트팜학과, 소프트웨어공학과, 컴퓨터인공지능학부, 생물산업기계공학과",
                    "sql_query": "SELECT 교과목명, 수업목표, 학년, 학기, 학과, 권장선수과목 FROM lectures_info WHERE 학과 IN ('스마트팜', '소프트웨어공학과', '컴퓨터인공지능학부', '생물산업기계공학과') AND 학년 IN (2, 3, 4);"
                },
                {
                    "question": "컴퓨터공학과 커리큘럼 알려줘. 관련학과: 컴퓨터인공지능학부",
                    "sql_query": "SELECT 학년, 학기, 교과목명, 수업목표, 권장선수과목 FROM info_curriculum WHERE 학과 = '컴퓨터인공지능학부' AND 학년 IN (2, 3, 4);"
                },
            ]

        # 프롬프트 템플릿 설정
        self.example_prompt = PromptTemplate.from_template(
            "User question: {question}\nSQL Query: {sql_query}\n"
        )

        # examples_text는 변수가 아니므로 템플릿에 직접 텍스트로 삽입
        if continuous == 1:
            self.examples_text = "\n\n".join([f"Example {i+1}: Question: {example['question']}\nHistory: {example.get('chat_memory', 'No history available')}\nSQL Query: {example['sql_query']}" for i, example in enumerate(self.examples)])
           
        else:
            self.examples_text = "\n\n".join([f"Example {i+1}: Question: {example['question']}\nSQL Query: {example['sql_query']}" for i, example in enumerate(self.examples)])

        if continuous == 1:
            self.few_shot_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt,
                prefix=(
                    "Given the following user question about curriculum data in the 'lectures_info' table, "
                    f"The table 'lectures_info' has the following columns and their data types:\n{self.table_info}\n"
                    "Utilize the following chat memory to help generate SQL queries with information such as department name or professor name:\n\n"
                    f"Generate a SQL query statement that combines the Chat memory and the question, like the following examples:\n\nExample text."
                    "Do not include any explanation or additional text, only return the SQL query.\n\n"

                ),
                suffix="User question: {question}\n Chat memory: {chat_memory} Example text: {examples_text}\nSQL Query:",
                input_variables=["question", "chat_memory","examples_text"]
            )
            
        else:
            # 새로운 대화에 대한 few_shot_prompt 설정
            self.few_shot_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt,
                prefix=(
                    "Given the following user question about curriculum data in the 'lectures_info' table, "
                    f"The table 'lectures_info' has the following columns and their data types:\n{self.table_info}\n"
                    f"The following examples should just be used as reference:\n\n{self.examples_text}."
                    "Do not include any explanation or additional text, only return the SQL query.\n\n"
                ),
                suffix="User question: {question}\nSQL Query:",
                input_variables=["question"]
            )

        # 체인 설정
        self.chain = RunnableSequence(
            RunnablePassthrough()
            | self.few_shot_prompt
            | self.llm
            | StrOutputParser()
        )
      
    async def allagent(self, question, continuous, history):
        self.set_examples_and_prompt(continuous)
        tmp_question = question + " 관련 학과: "
        relevant_department = self.recommend_relevant_department_all(question, None)
        department_names = [dept[0] for dept in relevant_department]
        department_str = ', '.join(department_names)
        tmp_question += f" {department_str}, "
        filtered_data = self.search_db(tmp_question, continuous, history)          
        if not filtered_data or filtered_data == "None" or filtered_data is None:
            return question, "None"
        else:
            return question, filtered_data

    def search_db(self, question, continuous, history):
        start_time = time.time()
        self.set_examples_and_prompt(continuous)  # chain 설정 보장
        if continuous == 0:
            sql_query = self.chain.invoke({"question": question}).strip()
        else:
            sql_query = self.chain.invoke({
                "question": question,
                "chat_memory": history,  
                "examples_text": self.examples_text
            }).strip()
        print(f"Question: {question}")
        print(f"Generated SQL Query: {sql_query}")

        try:
            cursor = DatabaseConnection.connection.cursor()
            # SQL 쿼리 실행
            cursor.execute(sql_query)
            # 쿼리에서 실제로 반환된 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]
            # 쿼리 결과 가져오기
            result = cursor.fetchall()
            # 결과를 딕셔너리 형식으로 변환하여 컬럼 이름과 값 매핑
            formatted_result = [
                {column: value for column, value in zip(column_names, row)}
                for row in result
            ]
            # 컬럼 이름과 변환된 결과 출력
            print("데이터 결과: ",formatted_result)
            end_time = time.time()
            print(f"sql query 생성 및 실행 시간: {(end_time - start_time) * 1000:.2f} ms")
            return formatted_result
          
        except Exception as e:
            print(f"쿼리 실행 에러: {sql_query}\n에러: {str(e)}")
            end_time = time.time()
            print(f"sql query 생성 및 실행 시간: {(end_time - start_time) * 1000:.2f} ms")
            return []

    async def searchagent(self, question, continuous, history):
        response = ""
        result = await self.allagent(question, continuous, history)
        if result and result[1] is not None and result[1] != "None" and result[1] != "None.":
            question_content, filtered_data = result
            async for partial_response in self.generate_response(question_content, filtered_data, continuous, history):
                response += partial_response
                yield partial_response        
        else:
            response = "해당 정보가 없습니다. 다시 확인해주세요."  # 또는 원하는 기본 응답
            yield response
        print(f"Question: {question}")
        print(f"Response: {response}\n")

    async def directagent(self, question, continuous, history):
        filtered_data = self.search_db(question, continuous, history)
        if not filtered_data or filtered_data == "None" or filtered_data is None:
            yield "No result"
        else:
            response = ""
            async for partial_response in self.generate_response(question, filtered_data, continuous, history):
                response += partial_response
                yield partial_response        
            print(f"Question: {question}")
            print(f"Response: {response}\n")
            



    '''def recommend_relevant_department(self, question, key_skills=None):
        input_text = question  
        # key_skills가 존재하면 question과 key_skills를 결합하여 입력 문장을 구성
        if key_skills:
            input_text += " " + key_skills
        question_embedding = setting.get_siamese_embedding(input_text)
        question_embedding = np.array(question_embedding)
        question_embedding = normalize(question_embedding.reshape(1, -1), axis=1) 
        D, I = setting.index.search(question_embedding.astype(np.float32), k=5)  # 상위 5개 유사 학과 검색
        recommended_departments = [(setting.departments_list[i], D[0][idx]) for idx, i in enumerate(I[0]) if D[0][idx] >= 0.81]
        return recommended_departments[:5]  # 상위 5개 학과만 반환'''
    async def recommend_relevant_department(self, question, key_skills=None):
        start_time = time.time()
        input_text = question  
        # key_skills가 존재하면 question과 key_skills를 결합하여 입력 문장을 구성
        if key_skills:
            input_text += " " + key_skills
        question_embedding = setting.get_siamese_embedding(input_text)
        question_embedding = np.array(question_embedding)
        question_embedding = normalize(question_embedding.reshape(1, -1), axis=1) 
        D, I = setting.index.search(question_embedding.astype(np.float32), k=5)  # 상위 5개 유사 학과 검색
        recommended_departments = [(setting.departments_list[i], D[0][idx]) for idx, i in enumerate(I[0]) if D[0][idx] >= 0.81]
        end_time = time.time()
        print(f"관련 학과 검색 시간: {(end_time - start_time) * 1000:.2f} ms")
        rec_dep =  recommended_departments[:4]  # 상위 5개 학과만 반환
        for chunk in rec_dep:  # 리스트의 각 항목(chunk)
            for char in chunk:  # 항목을 한 글자씩 반복
                yield char  # 한 글자씩 반환
    def recommend_relevant_department_all(self, question, key_skills=None):
        start_time = time.time()
        input_text = question  
        # key_skills가 존재하면 question과 key_skills를 결합하여 입력 문장을 구성
        if key_skills:
            input_text += " " + key_skills
        question_embedding = setting.get_siamese_embedding(input_text)
        question_embedding = np.array(question_embedding)
        question_embedding = normalize(question_embedding.reshape(1, -1), axis=1) 
        D, I = setting.index.search(question_embedding.astype(np.float32), k=5)  # 상위 5개 유사 학과 검색
        recommended_departments = [(setting.departments_list[i], D[0][idx]) for idx, i in enumerate(I[0]) if D[0][idx] >= 0.81]
        end_time = time.time()
        print(f"관련 학과 검색 시간: {(end_time - start_time) * 1000:.2f} ms")
        return recommended_departments[:1]  # 상위 5개 학과만 반환
    # async def recommend_relevant_department(self, question, key_skills=None):
    #     start_time = time.time()
    #     input_text = question
    #     if key_skills:
    #         input_text += " " + key_skills

    #     # LLM을 사용하여 관련 학과 추천
    #     prompt = f"""
    #     Input: "{input_text}"
    #     The following input is a job description, related question, or department name. 
    #     Based on the following input, please recommend relevant academic departments name from those listed in {setting.departments_list} where this topic can be studied.But under NO circumstances should you include any department or course that is not listed in {setting.departments_list}.
    #     Do not provide any additional explanations or details.

    #     Remember, recommendations must be strictly from the provided list {setting.departments_list}.
    #     """
    #     # OpenAI API를 사용하여 LLM 호출
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4o-mini",  
    #         messages=[
    #             {"role": "system", "content": "You are an assistant that helps recommend academic departments based on input."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=500, 
    #         n=1,
    #         stop=None,
    #         temperature=0.5,
    #         stream=True,
    #         seed = 1234
    #     )
    #     response_content = ""
    #     # 스트리밍된 응답을 비동기적으로 처리
    #     for chunk in response:
    #         content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
    #         if content_chunk:
    #             response_content += content_chunk
    #             yield content_chunk  # 한 글자씩 반환

    #     end_time = time.time()
    #     print(f"관련 학과 검색 시간: {(end_time - start_time) * 1000:.2f} ms")
    # def recommend_relevant_department_all(self, question, key_skills=None):
    #     start_time = time.time()
    #     input_text = question
    #     if key_skills:
    #         input_text += " " + key_skills

    #     # LLM을 사용하여 관련 학과 추천
    #     prompt = f"""
    #     Input: "{input_text}"
    #     The following input is a job description, related question, or department name. 
    #     Based on the following input, please recommend relevant academic departments from those listed in {setting.departments_list} where this topic can be studied.But under NO circumstances should you include any department or course that is not listed in {setting.departments_list}.
    #     Do not provide any additional explanations or details.

    #     Remember, recommendations must be strictly from the provided list {setting.departments_list}.
    #     """
    #     # OpenAI API를 사용하여 LLM 호출
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4o-mini",  
    #         messages=[
    #             {"role": "system", "content": "You are an assistant that helps recommend academic departments based on input."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=500, 
    #         n=1,
    #         stop=None,
    #         temperature=0,
    #     )
    #     response_text = response['choices'][0]['message']['content'].strip()
    #     print(response_text)

    #     seen_departments = set()  # 중복된 학과를 추적하기 위한 집합
    #     recommended_departments = []

    #     for line in response_text.split("\n"):
    #         line = line.strip()  # 공백 제거
    #         if line.startswith("-"):  # "-"로 시작하는 줄만 처리
    #             department = line[1:].strip()  # "-" 뒤의 학과명만 추출

    #             # 중복 학과가 아니면 추가
    #             if department not in seen_departments:
    #                 recommended_departments.append(department)
    #                 seen_departments.add(department)
        
    #     if (len(recommended_departments))<5:
    #         return recommended_departments
    #     end_time = time.time()
    #     print(f"관련 학과 검색 시간: {(end_time - start_time) * 1000:.2f} ms")
    #     return recommended_departments[:5]  # 상위 5개 학과만 반환



