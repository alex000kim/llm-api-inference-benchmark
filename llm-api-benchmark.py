# Usage:
# python llm-api-benchmark.py --api_key <your_api_key> --base_url <your_base_url> --model <your_model> --num_requests <your_num_requests>

import os
import time
import statistics
import openai
from tqdm import tqdm


import argparse


def benchmark(api_key, base_url, model, num_requests):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    TEST_PROMPT = "Tell me a brief history of the world in 100 words."

    total_latencies = []
    ttft_list = []
    tps_list = []

    for i in tqdm(range(1, num_requests + 1)):
        start_time = time.perf_counter()
        token_count = 0
        first_token_time = None

        try:
            response = client.chat.completions.create(model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT}
            ],
            stream=True)
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += len(delta.content.split())
        except Exception as e:
            print(f"Request {i} failed: {e}")
            continue

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_latency
        streaming_time = (end_time - first_token_time) if first_token_time else 0
        tps = token_count / streaming_time if streaming_time > 0 else 0

        total_latencies.append(total_latency)
        ttft_list.append(ttft)
        tps_list.append(tps)

        print(f"Request {i}:")
        print(f"  Total Latency        : {total_latency:.3f} seconds")
        print(f"  Time to First Token  : {ttft:.3f} seconds")
        print(f"  Tokens Received      : {token_count}")
        print(f"  Token Throughput     : {tps:.3f} tokens/second")
        print()

    if total_latencies:
        print("\n--- Overall Statistics ---")
        print(f"Total Successful Requests: {len(total_latencies)}")
        print("\nLatency (seconds):")
        print(f"  Min: {min(total_latencies):.3f}, Max: {max(total_latencies):.3f}, "
              f"Avg: {statistics.mean(total_latencies):.3f}, Median: {statistics.median(total_latencies):.3f}")
        print("\nTime to First Token (seconds):")
        print(f"  Min: {min(ttft_list):.3f}, Max: {max(ttft_list):.3f}, "
              f"Avg: {statistics.mean(ttft_list):.3f}, Median: {statistics.median(ttft_list):.3f}")
        print("\nToken Throughput (tokens/second):")
        print(f"  Min: {min(tps_list):.3f}, Max: {max(tps_list):.3f}, "
              f"Avg: {statistics.mean(tps_list):.3f}, Median: {statistics.median(tps_list):.3f}")
    else:
        print("No successful requests to calculate statistics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM API latency with streaming metrics.")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="API key to use for benchmarking.")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_API_BASE"),
                        help="API base to use for benchmarking.")
    parser.add_argument("--model", type=str, default=None, required=True,
                        help="LLM model to use")
    parser.add_argument("--num_requests", type=int, default=3,
                        help="Number of requests to send for benchmarking.")
    args = parser.parse_args()

    benchmark(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        num_requests=args.num_requests
    )
