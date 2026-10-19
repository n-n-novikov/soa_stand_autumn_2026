"""Проверки неоднозначных ответов Payment и паузы при его отказе."""

import json
import unittest
from unittest.mock import patch

import httpx

from app import CircuitBreaker, charge


class PaymentTests(unittest.IsolatedAsyncioTestCase):
    async def call_charge(self, handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await charge(client, CircuitBreaker(), {
                "id": "order-1", "amount_cents": 150,
            })

    async def test_success(self):
        result = await self.call_charge(lambda request: httpx.Response(201))
        self.assertEqual(result, "paid")

    async def test_only_bank_decline_rejects_order(self):
        for code in (402, 500, 400, 422):
            with self.subTest(code=code):
                result = await self.call_charge(lambda request: httpx.Response(code))
                self.assertEqual(result, "rejected" if code == 402 else None)

    async def test_lost_response_with_captured_payment(self):
        def handler(request):
            if request.method == "POST":
                raise httpx.ReadError("Ответ потерян", request=request)
            self.assertEqual(request.url.params["order_id"], "order-1")
            return httpx.Response(200, json=[{"charge_id": "charge-1"}])

        self.assertEqual(await self.call_charge(handler), "paid")

    async def test_retry_keeps_same_key_and_body(self):
        payments = []

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=[])
            payments.append((request.headers["Idempotency-Key"], json.loads(request.content)))
            if len(payments) == 1:
                raise httpx.ReadTimeout("Нет ответа", request=request)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            breaker = CircuitBreaker()
            order = {"id": "order-1", "amount_cents": 150}
            self.assertIsNone(await charge(client, breaker, order))
            self.assertEqual(await charge(client, breaker, order), "paid")
        self.assertEqual(len(payments), 2)
        self.assertEqual(payments[0], payments[1])
        self.assertEqual(payments[0][0], "order-1")
        self.assertEqual(payments[0][1]["amount_cents"], 150)

    async def test_unavailable_payment_and_lookup_stay_pending(self):
        def handler(request):
            raise httpx.ConnectError("Недоступен", request=request)

        self.assertIsNone(await self.call_charge(handler))

    async def test_open_breaker_does_not_send_requests(self):
        def handler(request):
            self.fail("При открытом breaker запросов быть не должно")

        breaker = CircuitBreaker()
        for _ in range(3):
            breaker.failure()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.assertIsNone(await charge(client, breaker, {"id": "order-1", "amount_cents": 150}))

    async def test_server_errors_do_not_open_shared_breaker(self):
        breaker = CircuitBreaker()
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(500)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for _ in range(5):
                self.assertIsNone(await charge(client, breaker, {"id": "order-1", "amount_cents": 150}))
        self.assertEqual(len(requests), 5)
        self.assertTrue(breaker.allow())


class CircuitBreakerTests(unittest.TestCase):
    def test_one_probe_after_pause_and_recovery(self):
        breaker = CircuitBreaker()
        with patch("app.time.monotonic", return_value=10):
            for _ in range(3):
                self.assertTrue(breaker.allow())
                breaker.failure()
            self.assertFalse(breaker.allow())
        with patch("app.time.monotonic", return_value=16):
            self.assertTrue(breaker.allow())
            self.assertFalse(breaker.allow())
            breaker.success()
            self.assertTrue(breaker.allow())

    def test_failed_probe_starts_new_pause(self):
        breaker = CircuitBreaker()
        with patch("app.time.monotonic", return_value=10):
            for _ in range(3):
                breaker.failure()
        with patch("app.time.monotonic", return_value=16):
            self.assertTrue(breaker.allow())
            breaker.failure()
            self.assertFalse(breaker.allow())
        with patch("app.time.monotonic", return_value=22):
            self.assertTrue(breaker.allow())


if __name__ == "__main__":
    unittest.main()
