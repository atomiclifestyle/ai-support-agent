## Decision log:

- Fine-tuned Encoder-only BERT for intent classification instead of relying on LLMs to minimize latency (<30ms) and API costs.  
- Implemented multi-factor escalation (BERT confidence + RAG retrieval score + sentiment threshold) rather than relying on a single confidence value.
- Fine-tuned BERT in two stages - Pre-trained with the banking77 datset to learn intent structure and further fine-tuned on AppleSupport's clustered intents.
- Filtered out historic customer handles/URLs during text preprocessing to prevent vector embedding pollution.
- Used cosine similarity with a fixed distance threshold for RAG retrieval to trigger fallback escalation when no close match exists.
- Formulated LLM-as-judge prompts with chain-of-thought evaluation before outputting numerical scores to increase human-judge alignment.  
- Subsampled evaluation batches for CLI runs to guarantee execution within the mandatory 15-minute time window.  
- Configured asynchronous batch processing (asyncio) for RAG generation and judge steps to prevent API rate-limit bottlenecks.
- Selected a single specific brand (e.g., @AppleSupport) rather than training across all brands to maintain high vector search domain specificity.  
- Stored conversation history as flattened text blocks in the prompt context instead of managing complex graph states for historical threads.