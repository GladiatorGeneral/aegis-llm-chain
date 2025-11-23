# TGI (Text Generation Inference) Dockerfile
FROM ghcr.io/huggingface/text-generation-inference:latest

# Environment variables for model configuration
ENV MODEL_ID=meta-llama/Llama-2-7b-hf
ENV NUM_SHARD=1
ENV MAX_INPUT_LENGTH=2048
ENV MAX_TOTAL_TOKENS=4096

# Expose ports
EXPOSE 8080

# Command will be overridden by docker-compose or K8s
