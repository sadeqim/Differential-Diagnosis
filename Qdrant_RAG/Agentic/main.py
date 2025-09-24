# main.py
import os
import json
import asyncio
from ollama import AsyncClient as AsyncOllamaClient
from qdrant_client import AsyncQdrantClient
from typing import Dict, Any

from rag_core import RAGChatbot 

MEDICAL_GEMMA_MODEL = "jwang580/medgemma_27b_text_it:latest" 
CRITIQUE_MODEL = "jwang580/medgemma_27b_text_it:latest" 
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CONFIDENCE_THRESHOLD = 7.5 # Hyperparameter: Tune based on evaluation

class MedicalAgent:
    """
    An agentic system using the Pre-Act method to answer medical questions.
    It first attempts a direct answer, critiques its own confidence, and
    then decides whether to use a RAG system as a fallback.
    """
    def __init__(self, ollama_client: AsyncOllamaClient, rag_chatbot: RAGChatbot):
        self.ollama_client = ollama_client
        self.rag_chatbot = rag_chatbot
        self.direct_answer_model = MEDICAL_GEMMA_MODEL
        self.critique_model = CRITIQUE_MODEL

    async def _generate_direct_answer(self, question: str) -> str:
        """Phase 1: Generates the initial answer from the fine-tuned model."""
        print("INFO: Generating direct answer...")
        response = await self.ollama_client.chat(
            model=self.direct_answer_model,
            messages=[{'role': 'user', 'content': question}]
        )
        return response['message']['content']

    async def _evaluate_confidence(self, question: str, answer: str) -> Dict[str, Any]:
        """Phase 2: Performs self-critique to get a structured confidence score."""
        print("INFO: Evaluating confidence of the direct answer...")
        prompt = f"""
        You are a meticulous medical expert AI evaluator. Your task is to assess the confidence of a generated answer to a given medical question. Provide your evaluation in a JSON format.

        The JSON object must contain two keys:
        1. "confidence_score": A float between 0.0 and 10.0, where 10.0 is absolute certainty and 0.0 is complete uncertainty.
        2. "reasoning": A brief, one-sentence explanation for your score.

        **Question:** "{question}"
        **Generated Answer:** "{answer}"

        **Evaluation JSON:**
        """
        
        response = await self.ollama_client.chat(
            model=self.critique_model,
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Force JSON output
        )
        
        try:
            # Clean up potential markdown code blocks around the JSON
            raw_json = response['message']['content'].strip().replace('```json', '').replace('```', '')
            evaluation = json.loads(raw_json)
            # Basic validation
            if 'confidence_score' in evaluation and 'reasoning' in evaluation:
                print(f"INFO: Confidence assessed: {evaluation['confidence_score']}/10. Reason: {evaluation['reasoning']}")
                return evaluation
            else:
                raise ValueError("Invalid JSON structure from critique model.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Could not parse confidence evaluation: {e}")
            # Default to low confidence on parsing failure to be safe
            return {"confidence_score": 0.0, "reasoning": "Failed to perform self-critique."}

    async def _invoke_rag(self, question: str):
        """Invokes the RAG system and streams the response."""
        print("INFO: Low confidence. Invoking RAG system as fallback.")
        async for chunk in self.rag_chatbot.stream_rag_response(question):
            yield chunk

    async def handle_question(self, question: str):
        """
        Orchestrates the full Pre-Act pipeline for a user question.
        This function streams the final, validated response.
        """
        print(f"\n--- New Query Received: '{question}' ---")
        # Phase 1: Initial Plan - Generate Direct Answer
        direct_answer = await self._generate_direct_answer(question)
        
        # Phase 2: Pre-Act - Self-Critique
        evaluation = await self._evaluate_confidence(question, direct_answer)
        confidence_score = evaluation.get('confidence_score', 0.0)
        
        # Phase 3: Decision Gate and Action
        if confidence_score >= CONFIDENCE_THRESHOLD:
            print("INFO: High confidence. Delivering direct answer.")
            # Yield the answer in the same dictionary format as RAG for consistency
            yield {"type": "llm_chunk", "data": direct_answer}
            yield {"type": "sources", "data": ["Internal Knowledge (Fine-tuned Model)"]}
        else:
            print("INFO: Low confidence. Switching to RAG system.")
            async for rag_chunk in self._invoke_rag(question):
                yield rag_chunk
        
        print("--- Query Handling Complete ---")


async def main():
    """Main entry point to run the agent."""
    # Initialize clients
    ollama_client = AsyncOllamaClient(host=OLLAMA_HOST)
    # NOTE: Ensure Qdrant is running and accessible
    qdrant_client = AsyncQdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
    
    rag_system = RAGChatbot(ollama_client=ollama_client, qdrant_client=qdrant_client)
    agent = MedicalAgent(ollama_client=ollama_client, rag_chatbot=rag_system)
    
    # Example questions
    questions = [
        # "What are the primary symptoms of myocardial infarction?", # Likely high confidence
        # "What was the conclusion of the 2024 Helsinki study on statin efficacy for non-binary patients over 50?", # Likely low confidence (too specific)
        # "what is love?" # Should trigger RAG and find nothing,
        "what is the medication and proper medicine for behchet disease?"
    ]
    
    for q in questions:
        final_answer = ""
        sources = []
        async for chunk in agent.handle_question(q):
            if chunk['type'] == 'llm_chunk':
                final_answer += chunk['data']
            elif chunk['type'] == 'sources':
                sources = chunk['data']
        
        print("\n--- FINAL RESPONSE ---")
        print(f"Question: {q}")
        print(f"Answer: {final_answer}")
        print(f"Sources: {sources}")
        print("-----------------------\n")


if __name__ == "__main__":
    asyncio.run(main())