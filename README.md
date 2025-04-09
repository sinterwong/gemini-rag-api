# gemini-rag-api
This project provides a REST API for interacting with Google's Gemini model, enhanced with Retrieval-Augmented Generation (RAG) capabilities. It allows users to query the Gemini model, leveraging external knowledge sources to improve the quality and relevance of the responses.

## Features

*   **Gemini Integration:** Seamlessly integrates with Google's Gemini model for text generation and understanding.
*   **RAG Support:** Implements Retrieval-Augmented Generation to enhance the model's responses with external data.
*   **REST API:** Provides a clean and easy-to-use REST API for interacting with the system.
*   **Customizable Knowledge Sources:** Supports various knowledge sources for RAG, allowing users to tailor the system to their specific needs.
*   **Scalable Architecture:** Designed with scalability in mind, allowing for efficient handling of large numbers of requests.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Google Cloud Project with Gemini API enabled
*   API Key for the Gemini API

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/sinterwong/gemini-rag-api.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd gemini-rag-api
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Set the environment variable for the Gemini API key:

    ```bash
    export GEMINI_API_KEY="your_api_key"

    export PYTHONPATH=${REPO_HOME}/app:${REPO_HOME}/app/services:${PYTHONPATH}
    ```

### Running the API

```bash
python app.py
```

This will start the API server on `http://127.0.0.1:9797`.

## Usage

### Adding document api
#### Request
```json
{
  "url": "http://127.0.0.1:9797/documents",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "documents": [
        {
            "doc_id": "uuid-xxxx",
            "text": "This is a document.", 
            "metadata": {
                "title": "xxx",
                "url": "xxx", 
                "id": "xxx"
                // ....
            }
        }
    ],
  }
}
```
#### Response
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "chunk_doc_ids": [
      "uuid-xxxx"
    ]
  }
}
```

### Querying API

#### Request
```json
{
  "url": "http://127.0.0.1:9797/query",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "query_text": "query text", 
    "top_k": 3
  }
}
```

#### Response
```json
{
    "results": [
        {
            "document": {
                "doc_id": "uuid-xxx",
                "metadata": {
                    "chunk_count": 0,
                    "chunk_index": 0,
                    "source_index": 0,
                    "source_text": "This is a document.",
                    
                    "title": "xxx",
                    "url": "xxx", 
                    "id": "xxx"
                    // ....
                },
                "text": "xxxxx",
                "score": 0.666
            }
        }
    ],
}

```


### Generating content api
#### Request
```json
{
  "url": "http://127.0.0.1:9797/generate",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "query_text": "prompt text",
    "top_k": 3,  // optional
    "include_sources": true // optinal
  }
}
```
#### Response
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "summary": "xxxx",
    "retrieved_documents": {
        "snippet": "xxxx",
        "metadata": {
            "title": "xxx",
            "url": "xxx", 
            "id": "xxx"
            // ....
        }
    }
  }
}
```

### Health Check api
#### Request
```json
{
  "url": "http://127.0.0.1:9797/health",
  "method": "GET",
  "headers": {
    "Content-Type": "application/json"
  }
}
```
#### Response
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "status": "ok", 
    "vector_store_stats": {
        "dimension": 3072, 
        "document_count": 0, 
        "index_type": "flat", 
        "is_trained": true,
        "metric_type": "l2"
    }
  }
}
```
