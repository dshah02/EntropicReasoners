import asyncio
import aiohttp
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_completion(session, question, temperature=1.0):
    """Asynchronously generates a completion for a given question with a specified temperature using the Claude API."""
    # Claude API endpoint and request structure
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01", # Use the latest API version
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514", # Or the specific Claude model you want to use
        "max_tokens": 1024, # Set an appropriate value for max tokens
        "temperature": temperature,
        "messages": [{"role": "user", "content": question}]
    }

    async with session.post(url, headers=headers, json=payload) as response:
        data = await response.json()
        # Extract the generated text, handling potential errors or different response structures
        if response.status == 200 and 'content' in data and data['content']:
            return data['content'][0]['text']
        else:
            print(f"Error generating completion for question: {question}. Status: {response.status}. Response: {data}")
            return "Error generating completion"


async def generate_completions_parallel(questions, num_completions_per_question=100, temperature=1.0):
    """Generates multiple completions for each question in parallel and groups them by question."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for question in questions:
            question_tasks = []
            for _ in range(num_completions_per_question):
                # Pass the temperature to the generate_completion function
                question_tasks.append(generate_completion(session, question, temperature=temperature))
            tasks.append(asyncio.gather(*question_tasks))
        return await asyncio.gather(*tasks)

def get_embeddings_by_question(completions_by_question, client):
    """
    Given a list of lists of completions (completions_by_question),
    returns a list of numpy arrays containing embeddings for each question's completions.
    """
    embeddings_by_question = []
    for question_completions in completions_by_question:
        try:
            result = [
                np.array(e.values) for e in client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=question_completions,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                ).embeddings
            ]
        except Exception as e:
            print(f"Error while fetching embeddings for question completions: {e}")
            result = []
        embeddings_by_question.append(np.array(result))
    return embeddings_by_question

# Assuming 'embeddings_by_question' is available from the previous step,
# where it's a list of NumPy arrays, each representing the embeddings for a question's completions.

def analyze_embedding_determinants(embeddings_by_question, num_samples=100, sample_size=10):
    """
    For each question's embeddings, randomly samples subsets and computes Gram matrix determinants.
    Prints and returns metrics for each question.
    """
    max_determinants = []
    results = []

    for i, embeddings_matrix in enumerate(embeddings_by_question):
        question_determinants = []

        for _ in range(num_samples):
            if embeddings_matrix.shape[0] >= sample_size:
                sampled_indices = np.random.choice(embeddings_matrix.shape[0], sample_size, replace=False)
                sampled_embeddings = embeddings_matrix[sampled_indices]
                G = np.dot(sampled_embeddings, sampled_embeddings.T)
                det_G = np.linalg.det(G)
                if det_G < 0:
                    det_G = 0
                question_determinants.append(det_G)
            else:
                print(f"Warning: Not enough embeddings ({embeddings_matrix.shape[0]}) to sample {sample_size} for Question {i+1}. Skipping sampling for this question.")
                break

        if question_determinants:
            question_max_determinant = np.sqrt(max(question_determinants))
            max_determinants.append(question_max_determinant)
            log_determinants = [np.log(det) for det in question_determinants if det > 0]
            average_log_determinant = np.mean(log_determinants) if log_determinants else float('-inf')
            average_determinant = np.mean(question_determinants)
            log_average_determinant = np.log(average_determinant) if average_determinant > 0 else float('-inf')

            # print(f"Question {i+1}:")
            # print(f"Log maximum determinant over {num_samples} samples of {sample_size} embeddings: {np.log(question_max_determinant) if question_max_determinant > 0 else float('-inf')}")
            # print(f"Average log determinant over {num_samples} samples of {sample_size} embeddings: {average_log_determinant}")
            # print(f"Log average determinant over {num_samples} samples of {sample_size} embeddings: {log_average_determinant}")

            results.append({
                "question_index": i,
                "max_determinant": question_max_determinant,
                "log_max_determinant": np.log(question_max_determinant) if question_max_determinant > 0 else float('-inf'),
                "average_log_determinant": average_log_determinant,
                "log_average_determinant": log_average_determinant,
                "determinants": question_determinants
            })
        else:
            max_determinants.append(0)
            print(f"Question {i+1}: No determinants were calculated due to insufficient embeddings.")
            results.append({
                "question_index": i,
                "max_determinant": 0,
                "log_max_determinant": float('-inf'),
                "average_log_determinant": float('-inf'),
                "log_average_determinant": float('-inf'),
                "determinants": []
            })

    return results, max_determinants


if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What are the main causes of climate change?"
    ]

    # Step 1: Generate completions
    loop = asyncio.get_event_loop()
    completions_by_question = loop.run_until_complete(
        generate_completions_parallel(questions, num_completions_per_question=5, temperature=0.7)
    )

    # Step 2: Get embeddings for completions
    embeddings_by_question = get_embeddings_by_question(completions_by_question, client)

    # Step 3: Analyze embedding determinants
    analysis_results, max_determinants = analyze_embedding_determinants(embeddings_by_question, num_samples=10, sample_size=3)

    # Print summary
    for i, result in enumerate(analysis_results):
        print(f"\nQuestion {i+1}: {questions[i]}")
        print(f"Max Determinant (sqrt): {result['max_determinant']}")
        print(f"Log Max Determinant: {result['log_max_determinant']}")
        print(f"Average Log Determinant: {result['average_log_determinant']}")
        print(f"Log Average Determinant: {result['log_average_determinant']}")