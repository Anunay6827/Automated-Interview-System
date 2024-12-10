from imports import *
from text_processing import *
from evaluation import *
from speech_input import *

# Run automatic interview with evaluation and grading based on audio input and predefined questions
def automatic_interview():
    visited = set()  
    evaluation_results = []  
    total_questions = 10
    user_intro = ""
    weighted_grades = []  

    intro_questions = [
        "Tell me about yourself.",
        "What are your key strengths?",
        "What is your educational background?"
    ]

    random.shuffle(intro_questions)  
    for i, question in enumerate(intro_questions):
        if i >= 3:  
            break
        print(f"Q{i+1}: {question}")
        answer = listen_to_audio()
        
        if answer:
            visited.add(question)  
            evaluation_results.append(evaluate_answer(answer, "", []))  
            user_intro += f" {answer}"  

    user_intro_keyphrases = user_intro.split()

    for i in range(len(intro_questions), total_questions):

        data['similarity'] = data['questions_keyphrases'].apply(
            lambda x: compute_cosine_similarity(user_intro_keyphrases, x)
        )

        unvisited_data = data[~data['questions'].isin(visited)]
        if unvisited_data.empty:
            print("No more unvisited questions available.")
            break

        most_similar_idx = unvisited_data['similarity'].idxmax()
        selected_question = data.loc[most_similar_idx, 'questions']
        selected_keyphrases = data.loc[most_similar_idx, 'answers_keyphrases']
        correct_answer = data.loc[most_similar_idx, 'answers']

        print(f"Q{i+1}: {selected_question}")
        answer = listen_to_audio()

        if answer:
            visited.add(selected_question)

            evaluation = evaluate_answer(answer, correct_answer, selected_keyphrases)

            print(f"Evaluation Result for Question {i+1}: {evaluation}")

            evaluation_results.append(evaluation)

            weighted_grade = calculate_weighted_grade(evaluation, answer)
            weighted_grades.append(weighted_grade)

            user_intro += f" {answer}"
            user_intro_keyphrases = user_intro.split()  

    print("\nInterview Completed. Evaluation Results:")

    for i, result in enumerate(evaluation_results):
        print(f"Q{i+1} - Metrics: {result}")
        if i < len(weighted_grades):  
            print(f"Q{i+1} - Weighted Grade: {weighted_grades[i]:.2f}")

    overall_score = sum(weighted_grades) / len(weighted_grades) if weighted_grades else 0
    overall_score_normalized = normalize_score(overall_score, 100)
    print(f"\nOverall Score: {overall_score_normalized/10:.2f}")

automatic_interview()