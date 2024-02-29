# Created by scalers.ai for Dell
"""
This script defines a FastAPI application for deploying and running a
vLLM (Very Large Language Model) engine. The application handles incoming
requests containing text prompts, max tokens and temperature and runs
inference on the vLLM async engine, and returns the generated text as responses.

The script is modified version of doc/source/serve/doc_code/vllm_example.py
from https://github.com/ray-project/ray/tree/master repo.
"""

import json
import os
from enum import Enum
from typing import AsyncGenerator, List, Optional

from fastapi import BackgroundTasks
from huggingface_hub import HfFolder
from pydantic import BaseModel
from ray import serve
from ray.serve import Application
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class DataTypes(str, Enum):
    """Supported datatypes by vLLM engine."""

    auto = "auto"
    fp32 = "float32"
    fp16 = "float16"
    bf16 = "bfloat16"
    half = "half"
    float = "float"


class DeployArgs(BaseModel):
    """DeployArgs pydantic model."""

    gpu_count: Optional[int] = 1
    model_name: Optional[str] = "meta-llama/Llama-2-7b-chat-hf"
    data_type: Optional[DataTypes] = DataTypes.bf16
    batch_size: Optional[int] = 32
    batch_wait_timeout_s: Optional[int] = 0.1
    hf_token: str


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 20},
    max_concurrent_queries=32,
)
class VLLMPredictDeployment:
    """vLLM engine based inference class."""

    def __init__(self, deploy_args: DeployArgs):
        """Initialize the VLLMPredictDeployment class."""
        self.batch_size = deploy_args.batch_size
        self.batch_wait_timeout_s = deploy_args.batch_wait_timeout_s
        HfFolder.save_token(os.environ.get("HF_TOKEN", deploy_args.hf_token))
        kwargs = {
            "model": deploy_args.model_name,
            "dtype": deploy_args.data_type,
            "tensor_parallel_size": deploy_args.gpu_count,
        }
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_results(
        self, results_generator
    ) -> AsyncGenerator[bytes, None]:
        """Stream inference results as sentences.

        :param results_generator: Generator for inference results.
        :type results_generator: AsyncGenerator
        :yields: JSON-encoded bytes containing text outputs.
        :rtype: AsyncGenerator
        """
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        """Abort request if the client is disconnected.

        :param request_id: ID of the request.
        :type request_id: str
        """
        await self.engine.abort(request_id)

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def infer_generate(self, requests: List[Request]) -> List[Response]:
        """Run inference on selected LLM model on vLLM engine.

        :param requests: List of HTTP requests.
        :type requests: List[Request]
        :returns: List of HTTP responses.
        :rtype: List[Response]
        """
        responses = []
        for request in requests:
            request_dict = await request.json()
            prompt = request_dict.pop("prompt")
            stream = request_dict.pop("stream", False)
            temperature = request_dict.pop("temperature", 1)
            max_tokens = request_dict.pop("max_tokens", 256)
            sampling_params = SamplingParams(
                **request_dict, max_tokens=max_tokens, temperature=temperature
            )
            request_id = random_uuid()
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id
            )
            if stream:
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.may_abort_request, request_id)
                responses.append(
                    StreamingResponse(
                        self.stream_results(results_generator),
                        background=background_tasks,
                    )
                )
            else:
                final_output = None
                async for request_output in results_generator:
                    if await request.is_disconnected():
                        await self.engine.abort(request_id)
                        responses.append(Response(status_code=499))
                    final_output = request_output

                assert final_output is not None
                prompt = final_output.prompt
                text_outputs = [
                    prompt + output.text for output in final_output.outputs
                ]
                ret = {"text": text_outputs}
                responses.append(Response(content=json.dumps(ret)))

        return responses

    async def __call__(self, requests: List[Request]) -> List[Response]:
        return await self.infer_generate(requests)

    def update_batch_params(self):
        """Update the server parameters through handler."""
        self.infer_generate.set_max_batch_size(self.batch_size)
        self.infer_generate.set_batch_wait_timeout_s(self.batch_wait_timeout_s)


def typed_app_builder(deploy_args: DeployArgs) -> Application:
    """Application builder to handle typed deployment arguments.

    :param deploy_args: Deployment arguments.
    :type deploy_args: DeployArgs
    :returns: Ray Serve Application
    :rtype: Application
    """
    return VLLMPredictDeployment.bind(deploy_args)
