import time

from tqdm.auto import tqdm
from rag_helper import RAGBase

from threading import Lock


def calc_price(usage, input_price=0.75, output_price=4.5):

    input_tokens  = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
    output_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages, input_price=0.75, output_price=4.5):
    total_cost = 0.0
    for usage in usages:
        cost = calc_price(usage, input_price, output_price)
        total_cost += cost["total_cost"]
    return total_cost


def llm_structured (
    client,instructions,user_prompt, output_type, model="gpt-5.4-mini",chat_completion=False) :
    
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    if chat_completion :
        response = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=output_type,
        )
        response_text_out = response.choices[0].message.parsed

    else :
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=output_type,
        )
        response_text_out = response.output_parsed

    return response_text_out, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gpt-5.4-mini",
    max_retries=3,
    chat_completion=False
):
    for attempt in range(max_retries):
        try:
            if chat_completion:
                return llm_structured(
                    client,
                    instructions,
                    user_prompt,
                    output_type,
                    model=model,
                    chat_completion=True
                )

            else:
                return llm_structured(
                    client,
                    instructions,
                    user_prompt,
                    output_type,
                    model=model,
                )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress_with_delay(pool, seq, f, delay=1.0):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
            
            # Inject a delay (in seconds) right after submitting a task
            if delay > 0:
                time.sleep(delay)

        for future in futures:
            result = future.result()
            results.append(result)

    return results

class RateLimiter:
    def __init__(self, requests_per_minute):
        self.lock = Lock()
        self.delay = 60.0 / requests_per_minute
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()