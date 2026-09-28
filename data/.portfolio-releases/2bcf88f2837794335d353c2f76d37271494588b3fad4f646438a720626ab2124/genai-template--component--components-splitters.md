# GenAI Template — components\-splitters Component

## Responsibilities

- The configured components are responsible for splitting documents into smaller, manageable chunks\. There are two main splitters: one that splits Markdown documents based on header hierarchy \(MarkdownDocumentSplitter\), and another that splits general documents into chunks based on sentence boundaries and token limits \(DocumentSplitter\)\. Both splitters preserve metadata and generate deterministic chunk IDs for traceability\.

Evidence:
- `src/genai_template/components/splitters/__init__.py`
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`

## Important Abstractions

- The key abstractions include:
- \- \*\*MarkdownDocumentSplitter\*\*: Uses LlamaIndex&\#x27;s MarkdownNodeParser to split Markdown documents into chunks based on headers, preserving header paths and metadata\.  
  \- \*\*DocumentSplitter\*\*: Uses LlamaIndex&\#x27;s SentenceSplitter to split documents into chunks constrained by chunk size and overlap, preserving metadata\.  
  \- \*\*DocumentChunk\*\*: Represents a chunk of a document with an ID, document ID, text content, and metadata\.  
  \- \*\*Document\*\*: Input document abstraction from LlamaIndex, containing text and metadata\.  
  \- \*\*Timer\*\*: Utility used to measure the time taken for splitting operations\.

Evidence:
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`

## Behavior

- \- \*\*MarkdownDocumentSplitter\*\* splits documents at Markdown headers, generating chunks with header\-path metadata that reflects the document&\#x27;s header hierarchy\. It handles fenced code blocks correctly by not splitting on Markdown\-like headers inside them\. It returns an empty list if input documents are empty or contain no text\.  
  \- \*\*DocumentSplitter\*\* splits documents into chunks based on sentence boundaries, respecting configured chunk size and overlap\. It logs chunk length statistics after splitting\. It returns an empty list if no documents are provided\.  
  \- Both splitters generate chunk IDs deterministically using the document ID and a zero\-padded index\.  
  \- Metadata from the original documents is preserved in the chunks\.  
  \- Embeddings are not set during splitting \(remain None\)\.

Evidence:
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `tests/components/splitters/test_markdown_splitter.py`
- `tests/components/splitters/test_sentence_splitter.py`

## Constraints

- \- The MarkdownDocumentSplitter relies on the header path separator to construct header paths, which can be customized but must be consistent\.  
  \- The DocumentSplitter depends on chunk size and overlap parameters to control chunk boundaries\.  
  \- Both splitters require input documents to have metadata fields like &quot;file\_name&quot; or &quot;doc\_id&quot; to generate chunk IDs\.  
  \- Empty documents or empty input lists result in no chunks\.  
  \- Markdown headers inside fenced code blocks do not trigger splits\.  
  \- Chunk IDs follow a strict pattern combining document ID and chunk index\.

Evidence:
- `src/genai_template/components/splitters/markdown_splitter.py`
- `src/genai_template/components/splitters/sentence_splitter.py`
- `tests/components/splitters/test_markdown_splitter.py`
- `tests/components/splitters/test_sentence_splitter.py`

## Testing Evidence

- \- Unit tests verify that the MarkdownDocumentSplitter correctly forwards the header path separator to the underlying parser\.  
  \- Tests confirm that header hierarchy and source metadata are preserved in Markdown chunks\.  
  \- Tests check that Markdown headers inside fenced code blocks do not cause splits\.  
  \- Tests verify that documents without headers remain a single chunk\.  
  \- Chunk IDs are tested for deterministic generation in both splitters\.  
  \- Empty input and empty document cases return empty chunk lists\.  
  \- DocumentSplitter tests confirm chunk size and overlap parameters are passed correctly\.  
  \- Metadata preservation and embedding absence are verified in both splitters\.  
  \- DocumentSplitter tests confirm splitting produces multiple chunks for large text\.

Evidence:
- `tests/components/splitters/test_markdown_splitter.py`
- `tests/components/splitters/test_sentence_splitter.py`
