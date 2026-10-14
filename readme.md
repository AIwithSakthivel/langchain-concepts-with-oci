# LangChain Concepts Cookbook with OCI

[oci-langchain-cookbook.ipynb](oci-langchain-cookbook.ipynb) demonstrates LangChain concepts through synthetic hospitality, financial-operations, and government-service examples. Chat uses `ChatOCIGenAI`; text and image embeddings use `OCIGenAIEmbeddings` from `langchain_oci`.

The notebook keeps its Markdown short. Each code cell names its scenario and context; this README explains setup, behavior, and limitations.

## Setup

1. Configure an OCI API-signing profile in `~/.oci/config` with access to Generative AI in your compartment. The notebook uses `API_KEY` authentication.
2. Copy `.env.example` to `.env` beside the notebook and replace the required dummy values with your OCI settings. Signing credentials belong in the OCI configuration file.
3. Start Jupyter from this directory, select your Python kernel, and run the dependency-installation cell. Restart the kernel if prompted, then continue in order.

```dotenv
OCI_COMPARTMENT_ID=your-compartment-ocid
OCI_SERVICE_ENDPOINT=your-generative-ai-inference-endpoint
# Optional; these values are the defaults.
OCI_CONFIG_PROFILE=DEFAULT
OCI_CONFIG_FILE=~/.oci/config
```

`OCI_COMPARTMENT_ID` and `OCI_SERVICE_ENDPOINT` are required. Existing environment variables take precedence over `.env`. The configuration cell checks that the notebook exists in the working directory and reports missing required settings without printing their values.

The installation cell specifies `langchain==1.4.5`, `langchain-oci==0.3.2`, `python-dotenv>=1,<2`, `numpy>=1.24,<3`, and `pydantic>=2.7,<3`. The examples themselves use the OCI chat model.

Model IDs are set directly in their cells. Check that your region and account support the selected models and the capabilities required by each example. The notebook's configured IDs are examples, not a guarantee of availability.

## Concepts and examples

### 1. Chat models: configuration and initialization

`ChatOCIGenAI` provides the shared `llm`. The cell loads `.env`, checks required settings, and builds `oci_settings` for reuse by chat and embedding clients. Change `chat_model_id` here to select a model with tool-calling and structured-output support for later examples.

### 2. Chat messages and conversation history

`llm.invoke()` receives a list of role/content pairs. A system message supplies fictional hotel policy, human messages represent guest questions, and AI messages provide example prior replies. Those prior AI replies are supplied history; the invocation generates the next assistant response. This demonstrates how a chat model receives conversation context rather than retaining it automatically between independent calls.

### 3. Streaming chat responses

`llm.stream()` yields response chunks. The cell prints nonempty content as it arrives, using a fictional municipal energy-program checklist as context. Streaming changes how the response is consumed; the supplied instructions still define what the assistant should explain.

### 4. Async invocation and streaming

`asyncio.gather()` runs independent `llm.ainvoke()` calls concurrently for two synthetic operations tickets. `llm.astream()` then streams a combined handoff. Jupyter supports top-level `await`, so the cell does not create another event loop with `asyncio.run()`. It closes the model's async client with `await llm.aclose()` after the requests.

### 5. Tools: definition, binding, and invocation

`@tool` turns typed Python functions into tools with descriptions and argument schemas. The reservation and guest-service tools read synthetic dictionaries using exact identifiers and return `found: False` with `status: "not_found"` for unknown IDs.

`llm.bind_tools()` exposes the tool schemas to the model. Binding alone does not execute Python functions: the model returns selections in `tool_calls`, and the cell dispatches recognized tool names through a lookup map. This example prints the results; the next example returns them to the model.

### 6. Tool calling: a bounded execution loop

The manual loop appends each model response to conversation history, executes only registered tools, and appends a `ToolMessage` for every result. Each message preserves the model's `tool_call_id`, associating the result with its request.

The loop finishes when the model returns a response without tool calls. Unknown tools raise an error, and a five-iteration limit raises an error if the model has not finished. The example illustrates the orchestration that an application must implement when handling tool calls directly.

### 7. Agents with create_agent

`create_agent()` manages tool execution and message history around the same OCI model and hotel tools. The invocation supplies a recursion limit of 12. This limit bounds graph execution; it is not the same unit as the manual loop's five iterations.

After execution, the cell inspects returned `ToolMessage` objects and asserts that both expected lookup tools ran. That assertion checks tool execution, not the accuracy of every sentence in the final response. This is a demonstration using synthetic records, not a production deployment.

### 8. Structured output with Pydantic

`llm.with_structured_output(ServiceIncident)` requests a result matching a Pydantic schema. Fields include the incident ID, an allowed priority value, affected channels, and whether staff review is required. The example prints the validated object's JSON representation.

Schema validation checks the output's structure and field types. It does not independently verify factual accuracy. The fictional report remains the source for the internal triage draft.

### 9. Multimodal chat: image input (optional)

A separate `vision_llm` receives a `HumanMessage` containing text and an image loaded with `load_image()`. It describes visible components and labeled connections without replacing the shared `llm`.

Set `vision_model_id` to an available vision-capable model and `vision_image_path` to a local diagram. The configured path is `sample_design_diagram.png`; relative paths resolve from the project root. Missing files raise an error. The prompt asks the model to report unreadable labels and avoid inferring security controls.

### 10. Text embeddings and semantic search (optional)

`OCIGenAIEmbeddings.embed_documents()` embeds four fictional procedure documents; `embed_query()` embeds a search question. NumPy computes cosine similarities and selects the document with the highest score.

Set `embedding_model_id` to an available text embedding model. Query and document vectors must come from the same model with compatible dimensions. The code rejects zero-length vectors because cosine similarity would be undefined.

This is an in-memory retrieval example. It does not use a vector database, apply a calibrated relevance threshold, or generate a grounded answer from the retrieved document. It always selects the highest-scoring document, even if every match is weak. The procedure thresholds, timelines, and compliance statements are fictional example text.

### 11. Multimodal embeddings: text-to-image search (optional)

`embed_image_batch()` embeds local diagrams, and `embed_query()` embeds a textual description. Cosine similarity ranks the images against the query in a shared vector space.

Set `image_embedding_model_id` to a model that supports both text and image embeddings in that space. The configured model is `cohere.embed-v4.0`; `image_paths_setting` contains comma-separated local paths to `sample_design_diagram.png` and `sample_sequence_diagram.png`. The example checks that files exist and rejects zero-length vectors. It makes one embedding request per image plus a text-query request.

As in text search, the highest score is a relative ranking, not a calibrated guarantee that the image matches the query.

## Sample diagrams

`sample_design_diagram.png` and `sample_sequence_diagram.png` are newly AI-generated, fictional, vendor-neutral teaching assets. They show a generic service portal, backend API, database, and notification service. They were generated from text descriptions without using the previous confidential diagrams as references and contain no Oracle or OCI branding or architecture.

The design diagram shows component connections; the sequence diagram shows request submission, persistence, notification acceptance, and confirmation to the user. Notification acceptance does not establish delivery.

## Running and validating

Run the notebook in order because later sections reuse configuration, tools, imports, and NumPy arrays defined earlier. Sections 9–11 are optional and require their own model capabilities; skip them when those models are unavailable.

Generated responses and tool selections may vary. The agent assertion verifies that both lookup tools executed. If a live request fails, check the selected model's capabilities, region access, authentication, and local image paths. Review generated text against the supplied synthetic facts before using it outside the demonstration.

The notebook retains example outputs and execution counts. Review any newly generated outputs before sharing. The included `.gitignore` excludes local environment files, Python environments, notebook checkpoints, the local `.oci` directory, and common signing-key formats while keeping `.env.example` publishable. Keep other OCI credential files and private images out of version control. The `.env.example` file contains dummy configuration values.

## References

- [Oracle LangChain OCI integration](https://github.com/oracle/langchain-oracle/tree/main/libs/oci)
- [LangChain create_agent reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [LangChain streaming documentation](https://docs.langchain.com/oss/python/langchain/streaming)
