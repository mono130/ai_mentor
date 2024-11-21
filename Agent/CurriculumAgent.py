
import numpy as np
import openai
import time
import AllAgent
from langchain.memory import ConversationBufferMemory

class CurriculumAgent(AllAgent.AllAgent):
    def __init__(self):
        super().__init__()
        self.memory = ConversationBufferMemory(return_messages=True)
        
    async def curriculumagent(self, question, continuous, history):
        start_time = time.time()
        key_skills = ""  # key_skills를 빈 문자열로 초기화
        department_names = ""
        response_curriculum = ""
        num_departments = 0
        # identify_key_skill이 async generator를 반환하는 경우, async for 사용
        async for part in self.identify_key_skill(question):
            key_skills += part  # 각 부분을 key_skills에 추가
            for char in part:
                yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt =  "\n\n해당 분야와 파악된 핵심 기술과 관련된 학과입니다.\n"
        for char in prompt:
            yield char  # 받은 부분을 즉시 한 글자씩 yield
        async for part in self.recommend_relevant_department(question, key_skills):
            if isinstance(part, (int, float, np.floating)):
                print(part)
                if num_departments <= 3:
                    department_names += ", "
                    yield ", "
                    
            else:
                department_names += part
                num_departments += 1
                print(len(department_names))
                for char in part:
                    yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt = "\n\n위에서 언급된 핵심 기술과 관련된 학과를 기반으로 해당 학기에 대한 추천 커리큘럼입니다.\n"
        for char in prompt:
            yield char
        # department_names = "회계학과, 농경제유통학부, 공공인재학부, 도시공학과, 통계학과, 환경공학과"
        # print_department_str = f"해당 분야와 파악된 핵심 기술과 관련된 학과는 {department_str}입니다"
        result = self.search_db(question + " 관련학과: " + department_names, continuous, history)
        async for part in self.generate_curriculum(question, department_names, key_skills, result, history):
            response_curriculum += part  # 각 부분을 key_skills에 추가
            for char in part:
                yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt = "이와 같이 추천된 강의들은 해당 분야에서 요구되는 핵심 기술과 지식을 체계적으로 습득할 수 있도록 구성되었습니다."
        for char in prompt:
            yield char
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def dep_curriculumagent(self, question, continuous, history):
        start_time = time.time()
        relevant_department = self.recommend_relevant_department_all(question, None)
        response_curriculum = ""
        result = self.search_db(question + " 관련학과: " + ", ".join(dept[0] for dept in relevant_department), continuous, history)
        async for part in self.generate_curriculum(question, relevant_department, None, result, history):
            response_curriculum += part  # 각 부분을 key_skills에 추가
            for char in part:
                yield char  # 받은 부분을 즉시 한 글자씩 yield
                
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def direct_curriculumagent(self, question, continuous, history):
        start_time = time.time()
        filtered_data = self.search_db(question, continuous, history)
        if not filtered_data or filtered_data == "None" or filtered_data is None:
            yield "No result"
        else:
            response_curriculum = ""
            async for part in self.generate_curriculum(question, None, None, filtered_data, history):
                response_curriculum += part  # 각 부분을 key_skills에 추가
                for char in part:
                    yield char  # 받은 부분을 즉시 한 글자씩 yield
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def identify_key_skill(self, question):
        start_time = time.time()
        prompt = f""" 
            "When a user expresses interest in a specific career, follow these steps:
            Career Definition: Define the career specified by the user in a concise and clear manner. The definition should include the primary role, responsibilities, and objectives in the field. 
            List and Output Required Technical Skills:
            Technical Skills: List the key technical skills required to succeed in this career. Include relevant technologies, tools, processes, and software.
            Output Format:
            진로 정의:
            [Career Name] is a professional who [Career Definition: Primary role, responsibilities, and key technologies used].
            필요한 기술적 정의:
            [Technology/Tool 1]
            [Technology/Tool 2]
            [Technology/Tool 3]
            [Technology/Tool 4]
            [Technology/Tool 5]
            Use this structure to provide relevant information based on the user's query."
            Example Outputs:
            User Input: "I want to become an AI Machine Failure Analysis Expert."
            Career Definition:
            An AI Machine Failure Analysis Expert specializes in utilizing artificial intelligence technologies to identify, diagnose, and propose solutions for mechanical failures. This role focuses on applying AI-based tools and algorithms to enhance the accuracy and efficiency of failure diagnostics and implement preventive measures.
            Technical Skills: The following technical skills are required:
            Proficiency in machine learning frameworks (e.g., TensorFlow, PyTorch)
            Data analysis and visualization tools (e.g., Python, MATLAB, Tableau)
            Knowledge of predictive maintenance systems and techniques
            Expertise in IoT-based monitoring systems
            Experience with mechanical simulation and modeling software (e.g., ANSYS, SolidWorks)
            user query :{question}
            Please print the output in Korean
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and kind assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.5,
            seed = 1234,
            stream=True  # 스트리밍 활성화
        )
        
        response_content = ""
        # 스트리밍된 응답을 비동기적으로 처리
        for chunk in response:
            content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
            if content_chunk:
                response_content += content_chunk
                yield content_chunk  # 한 글자씩 반환

        end_time = time.time()
        print(f"\n핵심 기술 파악 시간: {(end_time - start_time) * 1000:.2f} ms")
          
    async def generate_curriculum(self, question, department_list, key_skill, contexts_list, history):
        start_time = time.time()
        recommended_courses_set = set()
        self.conversation_question.append({question})
        categorized_courses = self.classify_courses(contexts_list)
        response_content = ""
        target_year = None
        for year in range(2, 5):  # 2학년부터 4학년까지 확인
            for semester in ["1학기", "2학기"]:
                flag = False
                year_semester_key = f"{year}학년 {semester}"
                if categorized_courses.get(year_semester_key):  # 해당 학기 데이터가 존재할 경우
                    target_year = year
                    flag = True
                    break  # 첫 번째로 발견된 학년으로 설정하고 반복 종료
            if flag is True:
                break
            if target_year is None:
                target_year = 2

        # 입력된 학년부터 4학년까지의 모든 학기 강의를 각각 처리
        for year in range(5, target_year-1, -1):
            for semester in ["2학기", "1학기"]:
                semester_start_time = time.time()
                response = ""
                year_semester_key = f"{year}학년 {semester}"
                relevant_courses = categorized_courses.get(year_semester_key, [])
                if relevant_courses:
                    count = 0
                    context_str = "\n\n".join([f"Context {i+1}: {str(ctx)}" for i, ctx in enumerate(relevant_courses)])
                    for i, _ in enumerate(relevant_courses):
                        count+=1
                    if key_skill is None:
                        prompt = f"""
                            Please list all lectures in the given context and ensure that no duplicate lectures are included. 
                            You MUST answer only using the following contexts:\n\n{context_str} in Korean.
                            Format for query and answer:\n
                            {year_semester_key}에 대한 모든 강의입니다.\n\n
                            - 교과목1 
                            - 교과목2 
                            - 교과목3 
                            - 교과목4 
                            - 교과목5
                            - 교과목6
                            ...\n
                        """                        
                    else:
                        prompt = f"""
                        You are an expert educational planner tasked with creating a curriculum for students in reverse semester order, starting from 4th-grade.
                        ### Requirements:
                        1. Curriculum in Reverse:
                        Recommended Course list from Previous Semester: {', '.join(recommended_courses_set)}
                        Please check the list of courses above and select only those that are not on the list.
                        2. Curriculum related to questions, key skills and '수업목표':
                        Review the ‘수업목표’ in {context_str}. Course recommendations must be based exclusively on the ‘수업목표’ provided in `{context_str}`. Identify and select lessons that specifically contribute to the development of competencies related to {key_skill}, align with the context of {question}, and effectively support the learning progression towards achieving {', '.join(recommended_courses_set)}.
                        When generating a curriculum in reverse order, prioritize selecting prerequisite courses for any courses already included in {', '.join(recommended_courses_set)}. The LLM must independently verify that each selected prerequisite course logically supports the foundational knowledge or skills necessary for the subsequent courses already generated. Avoid recommending courses that do not meet this criterion.
                        Ensure the recommendation process avoids subjective assumptions and aligns strictly with the course objectives and the defined learning progression.
                        3. Curriculum using {department_list}:
                        `{department_list}` contains a list of departments ranked by their similarity to the given question and key_skill, sorted in descending order of similarity. Make sure that courses in departments with higher similarity in `{department_list}` are prioritized in the curriculum.
                        For example, if `{department_list}` contains departments A, B, and C (with A being the most similar), the curriculum should 'prioritize' courses in `{context_str}` if they are included. Do not create a curriculum where courses are recommended only in departments with lower similarity (e.g. B, C) and not in departments with higher similarity (e.g. A). If there are no courses in departments with higher similarity, proceed to the next most similar department.
                        4. Number of Lectures:
                        You must distribute FIVE selected lectures per semester. For each semester, if {count} is less than FIVE, you must recommend ALL lectures in {context_str}.
                        5. No Duplication of Course Names:
                        Each ‘교과목명’ must appear only once in the entire curriculum, regardless of the department or professor it belongs to. Even if the same course name appears under different departments or professors, it must be treated as a duplicate and excluded from subsequent recommendations.
                        You must select courses whose course names do not overlap with the previously generated results {', '.join(recommended_courses_set)}.
                        6. Content used:
                        You MUST answer only using the following contents:{context_str}.
                        Do not generate any courses, departments, or data that are not explicitly present in `{context_str}`.
                        7. Output Language:
                        You MUST answer in Korean.
                        8. Output Format:
                        {year_semester_key}에 대한 추천 강의입니다.\n\n
                        - 교과목1 (학과)
                        - 교과목2 (학과)
                        - 교과목3 (학과)
                        - 교과목4 (학과)
                        - 교과목5 (학과)\n
                        ### Additional Notes:
                        - Treat courses with identical names (교과목명) as duplicates, even if their department or professor is different. Ensure no duplicate course names appear across any semesters.
                        You must strictly adhere to all the requirements provided above."""
                    ############################기존 프롬프트#########################
                    # prompt = f"""
                    #     You are currently recommending the curriculum in reverse, starting from the 4th year.
                    #     Review the ‘수업목표’ in the list of lessons provided and select lessons that related with {question}.
                    #     You MUST answer only using the following contexts:\n\n{context_str} in Korean.
                    #     We will distribute the selected lectures, with 5 lectures per semester. For each semester,
                    #     You must make sure to list each '교과목명' in other department only once and avoid including the same course multiple times.
                    #     Format for query and answer:\n
                    #     {year_semester_key}에 대한 추천 강의입니다\n\n
                    #     - 교과목1 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목2 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목3 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목4 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목5 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n\n
                    # """
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful and kind assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        n=1,
                        stop=None,
                        temperature=0.5,
                        seed = 1234,
                        stream = True
                    )
                            # 스트리밍된 응답을 비동기적으로 처리
                    for chunk in response:
                        content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
                        if content_chunk:
                            response_content += content_chunk
                            recommended_courses_set.add(content_chunk)
                            yield content_chunk  # 한 글자씩 반환
                    yield "\n\n"
                    semester_end_time = time.time()
                    print(f"{year_semester_key} 커리큘럼 구성 시간: {(semester_end_time - semester_start_time) * 1000:.2f} ms")
        end_time = time.time()
        print(f"커리큘럼 구성 시간: {(end_time - start_time) * 1000:.2f} ms")

    
    def classify_courses(self, contexts):
        """
        강의 정보를 학년과 학기에 따라 분류.
        """
        categorized_courses = {
            "1학년 1학기": [],
            "1학년 2학기": [],
            "2학년 1학기": [],
            "2학년 2학기": [],
            "3학년 1학기": [],
            "3학년 2학기": [],
            "4학년 1학기": [],
            "4학년 2학기": []
        }
        for context in contexts:
            course_details = self.parse_course_info(context)
            year_semester = f"{course_details.get('학년')}학년 {course_details.get('학기')}"
            if year_semester in categorized_courses:
                categorized_courses[year_semester].append(context)
        return categorized_courses
    
    def parse_course_info(self, course_info):
        """
        학년과 학기 정보를 추출하여 반환.
        """
        course_details = {}
        if isinstance(course_info, dict):
            course_details['학년'] = course_info.get('학년', None)
            course_details['학기'] = course_info.get('학기', None)
        return course_details