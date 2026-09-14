SINGLE_TURN_QA = """You are a helpful assistant.

Use the provided context to answer the user's question. Add an inline citation such
as [S1] immediately after every claim supported by a source. Use only source labels
that appear in the context, and do not generate a bibliography or sources section.

If the answer cannot be found in the context, say you don't know.
An "I don't know" response does not require a citation.

Context:

{context}

Question:

{query}

Answer:"""
