import os
import time
import random
import requests
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# ── Logging to Application Insights ──────────────────────────────
logger = logging.getLogger(__name__)
conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
if conn_str:
    logger.addHandler(AzureLogHandler(connection_string=conn_str))
logger.setLevel(logging.INFO)

# ── Config from environment variables ────────────────────────────
APIM_ENDPOINT         = os.environ.get("APIM_ENDPOINT")
APIM_KEY              = os.environ.get("APIM_SUBSCRIPTION_KEY")
AOAI_PRIMARY_URL      = os.environ.get("AOAI_PRIMARY_URL")
AOAI_PRIMARY_KEY      = os.environ.get("AOAI_PRIMARY_KEY")
AOAI_SECONDARY_URL    = os.environ.get("AOAI_SECONDARY_URL")
AOAI_SECONDARY_KEY    = os.environ.get("AOAI_SECONDARY_KEY")

# ── Token bucket throttler ────────────────────────────────────────
class TokenBucket:
    def __init__(self, rpm, tpm):
        self.rpm        = rpm
        self.tpm        = tpm
        self.requests   = rpm
        self.tokens     = tpm
        self.last_fill  = time.monotonic()

    def _refill(self):
        now     = time.monotonic()
        elapsed = now - self.last_fill
        if elapsed >= 60:
            self.requests  = self.rpm
            self.tokens    = self.tpm
            self.last_fill = now

    def acquire(self, estimated_tokens=500):
        while True:
            self._refill()
            if self.requests >= 1 and self.tokens >= estimated_tokens:
                self.requests -= 1
                self.tokens   -= estimated_tokens
                return
            time.sleep(1)

throttler = TokenBucket(
    rpm=int(os.environ.get("RPM_LIMIT", 25)),
    tpm=int(os.environ.get("TPM_LIMIT", 80000))
)

# ── Circuit breaker ───────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.failures   = 0
        self.threshold  = threshold
        self.timeout    = timeout
        self.opened_at  = None
        self.state      = "closed"   # closed / open / half-open

    def can_call(self):
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.monotonic() - self.opened_at > self.timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: allow one probe

    def success(self):
        self.failures = 0
        self.state    = "closed"

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.state     = "open"
            self.opened_at = time.monotonic()
            logger.warning(f"Circuit OPEN after {self.failures} failures")

primary_cb   = CircuitBreaker()
secondary_cb = CircuitBreaker()

# ── Call Azure OpenAI directly (bypassing APIM) ──────────────────
def call_openai(endpoint, api_key, prompt, retries=3):
    url     = f"{endpoint}openai/deployments/gpt-4o-primary/chat/completions?api-version=2024-02-01"
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body    = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 100}

    for attempt in range(retries):
        try:
            throttler.acquire()
            resp = requests.post(url, headers=headers, json=body, timeout=30)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                jitter = random.uniform(0, wait * 0.3)
                logger.warning(f"429 received. Waiting {wait + jitter:.1f}s")
                time.sleep(wait + jitter)

            else:
                logger.error(f"Unexpected status: {resp.status_code}")
                raise Exception(f"HTTP {resp.status_code}")

        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    raise Exception("Max retries exhausted")

# ── Fallback chain: primary → secondary ──────────────────────────
def call_with_fallback(prompt):
    # Try primary
    if primary_cb.can_call():
        try:
            result = call_openai(AOAI_PRIMARY_URL, AOAI_PRIMARY_KEY, prompt)
            primary_cb.success()
            logger.info("Response from PRIMARY (East US 2)")
            return result, "primary"
        except Exception as e:
            primary_cb.failure()
            logger.warning(f"Primary failed: {e}. Switching to secondary.")

    # Try secondary
    if secondary_cb.can_call():
        try:
            result = call_openai(AOAI_SECONDARY_URL, AOAI_SECONDARY_KEY, prompt)
            secondary_cb.success()
            logger.info("Response from SECONDARY (Sweden Central)")
            return result, "secondary"
        except Exception as e:
            secondary_cb.failure()
            logger.error(f"Secondary also failed: {e}")

    raise Exception("All providers failed")

# ── Main loop ─────────────────────────────────────────────────────
def main():
    logger.info("Worker started. Sending test prompts in loop.")
    prompts = [
        "Summarize: Azure API Management handles rate limiting.",
        "Summarize: Circuit breakers prevent cascade failures.",
        "Summarize: Token buckets smooth API traffic bursts.",
        "Summarize: Dead letter queues catch failed messages.",
        "Summarize: Redundancy across regions ensures high availability.",
    ]
    i = 0
    while True:
        prompt = prompts[i % len(prompts)]
        try:
            result, provider = call_with_fallback(prompt)
            content = result["choices"][0]["message"]["content"]
            logger.info(f"[{provider.upper()}] {content[:80]}")
            print(f"[{provider.upper()}] {content[:80]}")
        except Exception as e:
            logger.error(f"All providers failed: {e}")
            print(f"[ERROR] {e}")
        i += 1
        time.sleep(2)

if __name__ == "__main__":
    main()
