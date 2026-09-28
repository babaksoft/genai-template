# GenAI Template — rag\-services Component

## Responsibilities

- The configured component is responsible for managing the lifecycle of document corpora \(sources\) and their deterministic vector indexes\. It registers new corpus directories as sources, lists available and registered sources, retrieves source metadata by identifier, and rebuilds vector indexes for a given source and RAG configuration\. It ensures that the source directories are valid and immediate children of a configured root directory\. The component also manages concurrency for index rebuilds by using process\-local locks keyed by deterministic collection names derived from source IDs and configuration fingerprints\.

Evidence:
- `src/genai_template/services/source_service.py`

## Important Abstractions

- Key abstractions include:
- \- \*\*Source\*\*: Represents a registered document corpus with metadata including its name and directory path\.  
  \- \*\*RagConfig\*\*: Immutable configuration used for indexing and retrieval, influencing the deterministic index\.  
  \- \*\*Index Collection Name\*\*: A fixed\-length deterministic string derived by hashing the source ID and index configuration fingerprint, used to identify vector store collections\.  
  \- \*\*IndexingPipeline\*\*: A pipeline abstraction that handles document splitting, embedding, and storing into the vector store\.  
  \- \*\*VectorStore\*\*: Protocol representing the backend vector index store\.  
  \- \*\*Locks\*\*: Process\-local locks to serialize rebuilds of the same index collection\.

Evidence:
- `src/genai_template/services/source_service.py`

## Behavior

- The component behaves as follows:
- \- Lists candidate corpus directories that exist under the configured root but are not yet registered as sources\.  
  \- Registers a new source by validating the directory name, ensuring it is an immediate child directory, and persisting the source metadata\.  
  \- Retrieves a source by its unique identifier, raising errors if not found\.  
  \- Rebuilds the deterministic index for a source and RAG configuration by:  
  \- Validating the source and configuration\.  
  \- Deriving the deterministic collection name\.  
  \- Acquiring a lock to serialize rebuilds\.  
  \- Deleting the existing vector store collection\.  
  \- Running an indexing pipeline that splits documents, embeds chunks, and writes to the vector store\.  
  \- Returning indexing metrics including document and chunk counts and indexing duration\.  
  \- Ensures that the source directory path remains consistent with the registered metadata\.  
  \- Raises appropriate errors for invalid directory names, missing directories, or non\-directory paths\.

Evidence:
- `src/genai_template/services/source_service.py`

## Constraints

- \- The source directory must be an immediate child of the configured corpus root directory; directory names containing path separators or relative references are rejected\.  
  \- The source directory must exist and be a directory at registration and rebuild time\.  
  \- Source names must be unique; duplicate registrations are rejected\.  
  \- Index rebuilds for the same deterministic collection are serialized using process\-local locks to prevent concurrent rebuilds\.  
  \- The deterministic collection name is a fixed\-length SHA\-256 hash prefix to ensure backend\-safe naming\.  
  \- The component depends on a RAG configuration registry service to load and parse persisted configurations\.  
  \- The component does not itself build or populate indexes at registration time; index building is a separate explicit operation\.

Evidence:
- `src/genai_template/services/source_service.py`

## Testing Evidence

- Unit tests verify:
- \- Listing candidate directories excludes already registered sources\.  
  \- Registering a source persists only the directory metadata without building an index\.  
  \- Registration rejects invalid directory names \(e\.g\., outside corpus root\) and duplicates\.  
  \- The deterministic collection name is fixed\-length, deterministic, and changes with source ID or index configuration\.  
  \- Different RAG configurations that share index\-affecting settings produce the same collection name\.  
  \- Rebuilding an index loads the persisted configuration, deletes the existing collection, runs the indexing pipeline, and returns correct metrics\.  
  \- The rebuild lock is shared across service instances for the same collection name, ensuring serialization\.  
  \- Observability tests confirm that index rebuild emits nested OpenTelemetry spans with appropriate attributes and error handling\.

Evidence:
- `tests/services/test_indexing_observability.py`
- `tests/services/test_source_service.py`
