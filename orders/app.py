"""Orders: заказы, надёжная публикация событий и бонусы за доставку."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

DATABASE_URL = os.environ["DATABASE_URL"]
PAYMENT_URL = os.environ["PAYMENT_URL"].rstrip("/")
INVENTORY_URL = os.environ["INVENTORY_URL"].rstrip("/")
EVENTS_URL = os.environ["EVENTS_URL"].rstrip("/")
WORKERS = 4

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orders")

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    items JSONB NOT NULL,
    amount_cents BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    step TEXT NOT NULL DEFAULT 'reserve',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS orders_pending
    ON orders (next_attempt_at) WHERE status = 'pending';
CREATE TABLE IF NOT EXISTS outbox (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
    published BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (id) WHERE NOT published;
CREATE TABLE IF NOT EXISTS processed_events (
    id TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS bonuses (
    user_id TEXT PRIMARY KEY,
    points BIGINT NOT NULL
);
"""


async def db():
    return await psycopg.AsyncConnection.connect(
        DATABASE_URL, autocommit=True, row_factory=dict_row, connect_timeout=5,
    )


class Item(BaseModel):
    sku: str = Field(min_length=1, strict=True)
    qty: int = Field(ge=1, strict=True)
    price_cents: int = Field(ge=0, strict=True)


class OrderRequest(BaseModel):
    user_id: str = Field(min_length=1, strict=True)
    items: list[Item] = Field(min_length=1)


class DeliveryEvent(BaseModel):
    id: str = Field(min_length=1, strict=True)
    type: Literal["delivery.confirmed"]
    user_id: str = Field(min_length=1, strict=True)
    bonus_points: int = Field(ge=1, le=2**63 - 1, strict=True)


class CircuitBreaker:
    """После трёх сетевых сбоев — пауза 5 секунд и один пробный запрос."""

    def __init__(self):
        self.failures = 0
        self.retry_at = 0.0
        self.probing = False

    def allow(self):
        if not self.retry_at:
            return True
        if time.monotonic() < self.retry_at or self.probing:
            return False
        self.probing = True
        return True

    def success(self):
        self.failures = 0
        self.retry_at = 0.0
        self.probing = False

    def failure(self):
        self.failures += 1
        if self.failures >= 3 or self.probing:
            self.retry_at = time.monotonic() + 5
        self.probing = False


async def charge(client, breaker, order):
    if not breaker.allow():
        return None
    try:
        response = await client.post(
            f"{PAYMENT_URL}/payments",
            headers={"Idempotency-Key": order["id"]},
            json={
                "order_id": order["id"],
                "amount_cents": order["amount_cents"],
                "currency": "RUB",
            },
        )
        code = response.status_code
    except httpx.HTTPError:
        # Ответ мог потеряться уже после списания. Пустой список ещё не
        # доказывает отказ: исходный запрос мог не успеть выполниться.
        code = None
        try:
            response = await client.get(
                f"{PAYMENT_URL}/payments",
                params={"order_id": order["id"]}, timeout=1,
            )
            if response.status_code == 200 and response.json():
                code = 200
        except (httpx.HTTPError, ValueError):
            pass

    # Ответ 500 означает, что сервис доступен, но операция не выполнена.
    # Её повторит этот заказ; останавливать остальные заказы не нужно.
    if code is not None:
        breaker.success()
    else:
        breaker.failure()
    if code in (200, 201, 402):
        return "rejected" if code == 402 else "paid"
    log.warning("Заказ %s: Payment недоступен или ответил %s", order["id"], code)
    return None


async def mark_paid(conn, order_id):
    # Статус и обязанность опубликовать событие сохраняются вместе.
    # При падении либо зафиксируются обе записи, либо ни одна.
    async with conn.transaction():
        await conn.execute("UPDATE orders SET status = 'paid' WHERE id = %s", (order_id,))
        await conn.execute(
            "INSERT INTO outbox (id, order_id) VALUES (%s, %s) "
            "ON CONFLICT (order_id) DO NOTHING",
            (str(uuid.uuid4()), order_id),
        )


async def process_order(client, breaker, order):
    async with await db() as conn:
        if order["step"] == "reserve":
            response = await client.post(
                f"{INVENTORY_URL}/reservations",
                json={
                    "order_id": order["id"],
                    "items": [{"sku": i["sku"], "qty": i["qty"]} for i in order["items"]],
                },
            )
            response.raise_for_status()
            await conn.execute("UPDATE orders SET step = 'charge' WHERE id = %s", (order["id"],))
            order["step"] = "charge"

        if order["step"] == "charge":
            # Контракт Orders допускает бесплатные товары; Payment принимает
            # только положительную сумму. Для бесплатного заказа оплаты нет.
            result = "paid" if order["amount_cents"] == 0 else await charge(client, breaker, order)
            if result is None:
                return
            if result == "paid":
                await mark_paid(conn, order["id"])
                return
            # Запоминаем отказ до снятия резерва: если DELETE не удался,
            # следующая попытка только снимет резерв, не вызывая оплату.
            await conn.execute("UPDATE orders SET step = 'release' WHERE id = %s", (order["id"],))

        response = await client.delete(f"{INVENTORY_URL}/reservations/{order['id']}")
        response.raise_for_status()
        await conn.execute("UPDATE orders SET status = 'rejected' WHERE id = %s", (order["id"],))


async def worker(client, breaker, selection_lock, active):
    while True:
        order = None
        try:
            # В стенде одна копия сервиса. Блокировка нужна только при выборе:
            # сетевые вызовы разных заказов выполняются параллельно.
            async with selection_lock:
                async with await db() as conn:
                    cursor = await conn.execute(
                        "SELECT * FROM orders WHERE status = 'pending' "
                        "AND next_attempt_at <= now() AND id != ALL(%s) "
                        "ORDER BY next_attempt_at, created_at LIMIT 1",
                        (list(active),),
                    )
                    order = await cursor.fetchone()
                if order:
                    active.add(order["id"])

            if order:
                await process_order(client, breaker, order)
                # Пауза растёт до 8 секунд. Успешные заказы уже не pending.
                delay = min(2 ** min(order["attempts"], 3), 8)
                async with await db() as conn:
                    await conn.execute(
                        "UPDATE orders SET attempts = attempts + 1, "
                        "next_attempt_at = now() + %s * interval '1 second' "
                        "WHERE id = %s AND status = 'pending'",
                        (delay, order["id"]),
                    )
        except Exception:
            log.exception("Ошибка фоновой обработки заказа %s", order["id"] if order else "—")
            await asyncio.sleep(1)
        finally:
            if order:
                active.discard(order["id"])
        await asyncio.sleep(0.1)


async def publish_event(client, event):
    try:
        response = await client.post(f"{EVENTS_URL}/events", json={
            "id": event["id"], "type": "order.paid", "order_id": event["order_id"],
        })
        if response.status_code == 202:
            return True
        log.warning("Событие %s: шина ответила %s", event["id"], response.status_code)
    except httpx.HTTPError as exc:
        log.warning("Событие %s: нет ответа шины (%s)", event["id"], exc)
    return False


async def publisher(client):
    # Отправитель один, поэтому захватывать строки outbox не требуется.
    while True:
        try:
            async with await db() as conn:
                cursor = await conn.execute(
                    "SELECT id, order_id FROM outbox WHERE NOT published ORDER BY id LIMIT 1",
                )
                event = await cursor.fetchone()
                if event:
                    if await publish_event(client, event):
                        # Если ответ или эта запись потерялись, отправим ещё
                        # раз. id события берётся из базы и не меняется.
                        await conn.execute("UPDATE outbox SET published = TRUE WHERE id = %s", (event["id"],))
                    else:
                        await asyncio.sleep(2)
        except Exception:
            log.exception("Ошибка отправки события")
            await asyncio.sleep(2)
        await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app):
    async with await db() as conn:
        await conn.execute(SCHEMA)
    breaker = CircuitBreaker()
    selection_lock = asyncio.Lock()
    active = set()
    async with httpx.AsyncClient(timeout=3) as client:
        tasks = [
            asyncio.create_task(worker(client, breaker, selection_lock, active))
            for _ in range(WORKERS)
        ]
        tasks.append(asyncio.create_task(publisher(client)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # FastAPI по умолчанию возвращает 422, но контракт требует 400.
    return JSONResponse(status_code=400, content={"error": "bad_request"})


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/orders", status_code=201)
async def create_order(body: OrderRequest):
    order_id = str(uuid.uuid4())
    amount = sum(item.qty * item.price_cents for item in body.items)
    if amount > 2**63 - 1:
        raise HTTPException(400, "Сумма заказа слишком велика")
    async with await db() as conn:
        await conn.execute(
            "INSERT INTO orders (id, user_id, items, amount_cents) VALUES (%s, %s, %s, %s)",
            (order_id, body.user_id, Jsonb([item.model_dump() for item in body.items]), amount),
        )
    return {"id": order_id, "status": "pending"}


@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    async with await db() as conn:
        cursor = await conn.execute(
            "SELECT id, status, amount_cents FROM orders WHERE id = %s", (order_id,),
        )
        order = await cursor.fetchone()
    if order is None:
        raise HTTPException(404, "Заказ не найден")
    return order


@app.post("/webhooks/delivery")
async def delivery_confirmed(event: DeliveryEvent):
    async with await db() as conn:
        # Уникальный id защищает и от последовательных, и от одновременных
        # копий. Отметка о событии и начисление фиксируются одной транзакцией.
        async with conn.transaction():
            cursor = await conn.execute(
                "INSERT INTO processed_events (id) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id",
                (event.id,),
            )
            if await cursor.fetchone():
                await conn.execute(
                    "INSERT INTO bonuses (user_id, points) VALUES (%s, %s) "
                    "ON CONFLICT (user_id) DO UPDATE SET points = bonuses.points + EXCLUDED.points",
                    (event.user_id, event.bonus_points),
                )
    # Ответ только после commit. Если он потеряется, повтор уже не начислит бонусы.
    return {"status": "ok"}


@app.get("/users/{user_id}/bonus")
async def get_bonus(user_id: str):
    async with await db() as conn:
        cursor = await conn.execute("SELECT points FROM bonuses WHERE user_id = %s", (user_id,))
        row = await cursor.fetchone()
    return {"points": row["points"] if row else 0}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
