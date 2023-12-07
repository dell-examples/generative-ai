# Llama 2 Large Language Model

The [Llama 2](https://ai.meta.com/llama/) model was released in July 2023 by Meta AI. It was trained on a dataset created from publicly available sources, and, when released, it performed equal to, or better than, most open source models, some of which were much larger than it.

The Llama 2 model is an example of a Large Language Model (LLM). As the name suggests, these models tend to be very large (typically around at least a few billion parameters), and excel at tasks that involve understanding the relationships between words in a sentence.

The model is available in 3 versions: a 7 billion parameter version (which has a file size of around 13 GB), a 13 billion parameter version (which has a file size of around 26 GB), and a 70 billion parameter version (which has a file size of around 137 GB). A model with a smaller number of parameters will be faster to train and use, but might not be as accurate as a model with a larger number of parameters.

Along with these versions, there are many different flavours of the model, such as Llama Chat, which is trained specifically with a chatbot use case in mind, and Code Llama, which is trained with the use case of helping programmers while they are writing code.

## Table Of Contents
* [Why Llama2](#why-llama-2)
* [Technical Details](#technical-details)
* [Further Reading](#further-reading)

## Why Llama 2

The Llama 2 model has several reasons to be preferable in commercial use cases. It has an open-source licence, performs better than most similarly sized models, and Meta AI has acknowledged, or thoroughly taken steps to reduce, the potential bias and toxicity of the model (which is a byproduct of training the model on a real world dataset).

Furthermore, as it is an open source model, many open source libraries, like HuggingFace and GGML, have integrated the model into their tools. Thus, the model can be used by only running a few lines of code. Along with this, Meta also includes a responsible use guide, and code examples to deploy the model.


## Technical Details

The Llama 2 architecture is mostly the same as the Llama 1 architecture, which used a transformer architecture, RMSNorm for pre-normalization, SwiGLU activation, and rotary positional embeddings. The tokenizer was also the same as Llama 1, which used the bytepair encoding (BPE) algorithm from SentencePiece.

In contrast to Llama 1, the Llama 2 model uses auto-regressive transformers with performance improvements, such as more robust data cleaning, updating data mixes, 40% more tokens, double context length, and Grouped-Query Attention (GQA). Meta gave more weight to factual sources, and removed sources known to contain a lot of personal information.


Llama 2 chat is a fine-tuned version of Llama 2. The dataset that they used started with a publicly available annotated dataset, to which they added more SFT data, and high-quality examples from vendors. It was determined by Meta AI that a limited set of clean instruction tuning data can achieve good quality, and came to the conclusion that 27540 annotations was enough.

They later found that the outputs from a fine-tuned model were competitive with human annotations, so they reprioritized annotation effort to iterative RLHF (Reinforcement Learning from Human Feedback). They used 2 main RLHF algorithms: Proximal Policy Optimization (PPO), and Rejection Sampling.

For better usability, Meta AI developed Ghost Attention (GAtt), a method designed to allow better dialogue flow. In terms of safety, Meta AI used safety-specific data annotation and tuning, with red-teaming and iterative evaluations.


## Further Reading

* [Meta AI-Llama 2 Official Website](https://ai.meta.com/llama/)
* [Llama Research Paper](https://arxiv.org/abs/2302.13971)
* [What are Large Language Models (LLM)? | AWS Blog](https://aws.amazon.com/what-is/large-language-model/)
* [Llama 2 Model Hugging Face](https://huggingface.co/meta-llama)
