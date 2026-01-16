"""
Benchmark script for entity-api continuous batching performance.

Tests /predict and /batch/predict endpoints with varying concurrency levels.
"""
import aiohttp
import asyncio
import time
import json
import click
import os


async def stress_test_predict(index, url, text):
    """Test single predict endpoint."""
    json_data = {
        'text': text,
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/predict", json=json_data) as response:
            result = await response.json()
            elapsed = time.time() - start_time

    return {
        "response_time": elapsed,
        "entities_found": len(result.get('name', [])) + len(result.get('address', [])) + 
                         len(result.get('ic', [])) + len(result.get('phone', [])) + 
                         len(result.get('email', [])),
    }


async def stress_test_batch_predict(index, url, texts):
    """Test batch predict endpoint."""
    json_data = {
        'texts': texts,
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/batch/predict", json=json_data) as response:
            result = await response.json()
            elapsed = time.time() - start_time

    return {
        "response_time": elapsed,
        "num_texts": len(result.get('results', [])),
    }


async def run_stress_test_predict(url, texts, concurrency):
    """Run predict stress test with given concurrency."""
    tasks = [
        stress_test_predict(i, url, texts[i % len(texts)])
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    return results


async def run_stress_test_batch_predict(url, texts_batch, concurrency):
    """Run batch predict stress test with given concurrency."""
    tasks = [
        stress_test_batch_predict(i, url, texts_batch[i % len(texts_batch)])
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    return results


@click.command()
@click.option('--url', default='http://localhost:8000', help='The API base URL.')
@click.option('--endpoint', type=click.Choice(['predict', 'batch']), default='predict',
              help='Endpoint to benchmark (predict or batch).')
@click.option('--save', required=True, help='Save folder name.')
@click.option('--concurrency-list', default='10,20,30,40,50,60,70,80,90,100',
              help='Comma-separated list of concurrency values to test.')
def main(url, endpoint, save, concurrency_list):
    """Run benchmark tests for entity-api."""
    os.makedirs(save, exist_ok=True)
    concurrency_values = [int(c) for c in concurrency_list.split(',')]

    # Test texts with various entities
    test_texts = [
        "nama saya Ahmad dari Kuala Lumpur",
        "hubungi saya di 0123456789",
        "email test@gmail.com untuk maklumat lanjut",
        "IC saya 900101-14-5678",
        "saya tinggal di Johor Bahru",
        "nama Ali dari Penang, hubungi 0162587806",
        "alamat saya di Sungai Petani, email ali@test.com",
        "IC saya 900505-14-6789, hubungi 0198765432",
    ]

    # Batch texts for /predict endpoint
    batch_texts = [
        ["nama saya Ahmad", "dari Kuala Lumpur"],
        ["hubungi 0123456789", "email test@gmail.com"],
        ["IC 900101-14-5678", "tinggal di Johor Bahru"],
    ]

    async def run_all():
        for concurrency in concurrency_values:
            print(f"\nTesting {endpoint} endpoint with {concurrency} concurrency")
            
            if endpoint == 'predict':
                results = await run_stress_test_predict(url, test_texts, concurrency)
                response_times = [r["response_time"] for r in results]
                entities_found = sum(r["entities_found"] for r in results)
                
                results_data = {
                    "concurrency": concurrency,
                    "response_time_avg": sum(response_times) / len(response_times),
                    "response_time_min": min(response_times),
                    "response_time_max": max(response_times),
                    "response_time_p50": sorted(response_times)[len(response_times) // 2],
                    "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)],
                    "response_time_p99": sorted(response_times)[int(len(response_times) * 0.99)],
                    "total_entities_found": entities_found,
                    "avg_entities_per_request": entities_found / len(results),
                    "throughput": concurrency / (sum(response_times) / len(response_times)),
                }
            else:  # batch
                results = await run_stress_test_batch_predict(url, batch_texts, concurrency)
                response_times = [r["response_time"] for r in results]
                total_texts = sum(r["num_texts"] for r in results)
                
                results_data = {
                    "concurrency": concurrency,
                    "response_time_avg": sum(response_times) / len(response_times),
                    "response_time_min": min(response_times),
                    "response_time_max": max(response_times),
                    "response_time_p50": sorted(response_times)[len(response_times) // 2],
                    "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)],
                    "response_time_p99": sorted(response_times)[int(len(response_times) * 0.99)],
                    "total_texts_processed": total_texts,
                    "avg_texts_per_request": total_texts / len(results),
                    "throughput": concurrency / (sum(response_times) / len(response_times)),
                }

            print(f"  Avg response time: {results_data['response_time_avg']*1000:.1f}ms")
            print(f"  Throughput: {results_data['throughput']:.1f} req/s")

            with open(f"{save}/{endpoint}_{concurrency}concurrency.json", "w") as outfile:
                json.dump(results_data, outfile, indent=4)

    asyncio.run(run_all())


if __name__ == '__main__':
    main()
