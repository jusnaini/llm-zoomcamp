# LLM Zoomcamp: Course Work & Assignment Submissions

This repository houses my assignment submissions and project work for the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

The goal is to learn how to build, evaluate, and deploy production-ready Large Language Model (LLM) applications, focusing heavily on Retrieval-Augmented Generation (RAG) and AI engineering workflows.

## 🏗️ Course Progress & Deliverables

The repository is organized by module. I am actively working through the curriculum, adding source code and assignments as I complete them.

### ✅ [Module 01: Agentic RAG](./01-agentic-rag/)

* Built a foundational text search and Retrieval-Augmented Generation (RAG) workflow.
* Developed an AI agent setup capable of executing specific functions based on user queries.
* Configured local environment management using Python's `uv` tool.

### ✅ [Module 02: Vector Search](./02-vector-search/)

* Embedded text using a lightweight ONNX model (all-MiniLM-L6-v2) — no PyTorch required
* Built vector search from scratch with numpy dot products
* Used minsearch.VectorSearch for indexed semantic retrieval
* Compared keyword vs vector search and combined them with hybrid search (Reciprocal Rank Fusion)

### ✅ [Module 03: Workflow Orchestration with Kestra](./03-orchestration/)

* Explored context engineering — why generic AI assistants fail and how to ground them in current documentation
* Used Kestra's AI Copilot to generate and refine flows faster than building manually
* Compared RAG vs no-RAG responses and observed hallucination firsthand
* Built autonomous AI agents that make decisions and call tools dynamically
* Implemented multi-agent systems where specialized agents collaborate on complex tasks
* Learned best practices for cost, security, and observability in production AI workflows

###  ✅ [Module 04: Evaluation & Quality Control](./04-evaluation/)

* Generated a ground truth from existing dataset with LLM and structured the output with Pydantic model
* Evaluated keyword, vector, and hybrid search using Hit Rate and MRR
* Compared all three search methods on numbers rather than intuition
* Tuned the RRF `k` parameter for hybrid search and measured its impact on retrieval quality

### ⏳ Module 05: Monitoring & Observability — *Upcoming*

* *Planned:* Tracking live system performance, user feedback, and prompt metrics.
