# Senior AI Engineer Job Task
## Document-Based Question Answering System ( LLM-free extractive QA system )

You have been tasked with designing and implementing a knowledge base system. Users must be able to query a collection of documents and receive substantiated answers derived exclusively from the content of those documents.
Documents may be added, edited, or deleted over time. The system must keep its information up to date and prevent the use of deleted documents or outdated versions when generating answers.
Retrieval quality, answer substantiation, and accurate document change management are especially important.

## Requirements for this RAG system:
- Document processing: Support for PDF, TEXT, and Markdown formats, and preparation of their content for retrieval.
- Change management: Capability to add, edit, and delete documents; ensures that outdated content is not used as the basis for new responses.
- Relevant retrieval: Identification of content segments relevant to the user's query and their use in generating a response.
- Evidence-based responses: Responses generated solely from retrieved content, accompanied by traceable references to the supporting document and specific section.
- Handling insufficient information: Clearly stating the lack of information, refraining from generating unsupported responses, and preventing model hallucinations.
- API access: Accessibility via API without the need for a graphical user interface (GUI).

## Models available to you:
You have access to the following embedding models:
1. BGE-M3
2. Embedding-3-Small
3. Embedding-3-Large
4. Gemini-Embedding-001

{
`https://models-interview.arvancloudai.ir/v1`,
`Authorization: Bearer <token>`,
`GET /v1/models`,
`POST /v1/embeddings`,
`120 requests/minute`,`200000 Characters/request`
}

* You don't need to use all the models; choose the most appropriate one and explain your choice.

## Implementation and evaluation points:
- You are free to choose the programming language, architecture, and tools.
- Code readability, separation of concerns, error handling, and testing key components are important to us.
- The quality of the solution and the reasoning behind technical decisions matter more than the number of tools used or the complexity of the architecture.