# ICL vs. In-Place TTT for Long-Context Recall in Modern Transformers

## Motivation
Large language models operate under a "train then freeze" paradigm, relying entirely on their fixed context window to process new information at inference time. In-Context Learning (ICL) places this information directly in the prompt, but performance degrades as context length grows. In-Place Test-Time Training (In-Place TTT), Feng et al. offers an alternative: compress context into the model's weights at inference by treating MLP output projection matrices as adaptable fast weights updated via a next-token-prediction-aligned objective. 

We propose to implement and benchmark both approaches on the same model to characterize the tradeoff between memory compression quality and prompt-based recall.

## Approach 
Use Qwen3-0.6B (28 layers, 32K context window) as our base. We will implement a modified version of Qwen3-0.6B with In-Place TTT by replacing the output projection of select MLP blocks with a TTT-enabled module that performs chunk-wise gradient updates during the forward pass on the input document. The modification is a drop-in enhancement.

## Evaluation
The evaluation follows an NVIDIA RULER-inspired protocol:
- ICL baseline: Input text is prepended to the context. A fresh frozen model with the full context is loaded per question.
- In-Place TTT: The model processes the input text and updates its fast weights. A fresh frozen model (without input context) is loaded per question, relying solely on weight-compressed knowledge.

We will test across context lengths from 1K to 32K tokens on RULER-style tasks (single/multi-hop NIAH, variable tracking, QA). Metrics: answer accuracy, GPU memory usage, and inference latency, each measured as a function of context length.

NOTE: We could also just follow the approach of using NVIDIA RULER  in the original paper

## Visualizations
Accuracy vs. context length (per task type)
Memory usage vs. context length
Inference latency vs. context length
Needle position × accuracy heatmap

## Tech Stack
PyTorch, HuggingFace Transformers, NVIDIA RULER, Datasets, W&B

## Expected Outcomes
We expect ICL to outperform In-Place TTT at short contexts, with In-Place TTT becoming competitive or superior as context length increases beyond the model's effective attention range, consistent with the significant RULER gains reported in the original paper. If In-Place TTT degrades at short contexts, that crossover point is itself a key finding.

## Class Information
Chang Min and Hung Ngo
CSCI357 (Spring 2026) - AI with Neural Nets
Professor Brian King
April 21, 2026