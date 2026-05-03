"""Run local validation scenarios against running bot."""

from __future__ import annotations

import json
from urllib import request as urlrequest

BASE_URL = "http://127.0.0.1:8080"


def post(path: str, payload: dict) -> dict:
    req = urlrequest.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urlrequest.urlopen(req, timeout=10).read().decode("utf-8"))


def get(path: str) -> dict:
    return json.loads(urlrequest.urlopen(BASE_URL + path, timeout=10).read().decode("utf-8"))


def main() -> None:
    print("healthz:", get("/v1/healthz"))
    print("metadata keys:", sorted(get("/v1/metadata").keys()))

    base_reply = {
        "conversation_id": "conv_check_1",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "received_at": "2026-05-02T00:00:00Z",
        "turn_number": 2,
    }
    print("intent:", post("/v1/reply", {**base_reply, "message": "Ok lets do it. Whats next?"}))
    print("off_topic:", post("/v1/reply", {**base_reply, "turn_number": 3, "message": "help me with GST filing"}))
    print("hostile:", post("/v1/reply", {**base_reply, "turn_number": 4, "message": "Stop messaging me. spam"}))

    auto = {
        "conversation_id": "conv_check_auto",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Thank you for contacting us! Our team will respond shortly.",
        "received_at": "2026-05-02T00:00:00Z",
        "turn_number": 2,
    }
    print("auto1:", post("/v1/reply", auto))
    print("auto2:", post("/v1/reply", {**auto, "turn_number": 3}))
    print("auto3:", post("/v1/reply", {**auto, "turn_number": 4}))


if __name__ == "__main__":
    main()

