"""
Prompts for QA System
All AI prompts centralized in one place
"""

# Prompt for QA processing
QA_PROMPT_TEMPLATE = """You are an AI assistant analyzing member messages to answer questions accurately.

Based on the following member messages, please answer the question. 
Be specific and extract exact information from the messages when available.
If the information is not available in the messages, say so clearly.

MEMBER MESSAGES:
{context}

QUESTION: {question}

Instructions:
1. Answer based ONLY on the information provided in the member messages
2. Be concise and direct
3. If asking about quantities (how many), provide a specific number
4. If asking about dates/times, provide specific dates mentioned
5. If asking about favorites or preferences, list the specific items mentioned
6. If the information is not in the messages, respond with "I don't have that information in the member messages."

ANSWER:"""

