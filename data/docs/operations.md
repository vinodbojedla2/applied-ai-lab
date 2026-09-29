# Operating an AI service

An AI service should record latency, errors, retrieval scores, token usage, and user feedback without logging secrets or sensitive document content. Monitor performance by query type and dataset version. Roll out prompt, index, and model changes behind versioned configuration, test a canary, and keep a rollback path.

Prompt injection can appear inside retrieved documents. Treat retrieved content as untrusted data rather than system instructions. Scope access before retrieval, do not expose private documents across users, and validate citations and outputs. A local demo with public sample documents is not a substitute for production authorization controls.
