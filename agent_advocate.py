#####################
# Agent Advocate
# Streamlit app for simulating multi-agent conversations with LLMs
#####################
# This app was AI generated to simulate political scenarios and 
# demo a multi-agent conversation where one LLM advocates for a oppositional 
# or minority view. 


from __future__ import annotations

import os
import re
from typing import Literal

import requests
import streamlit as st

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

DEFAULT_EXPERT_MODEL = os.getenv("EXPERT_MODEL", "phi3:mini")
DEFAULT_ADVOCATE_MODEL = os.getenv("ADVOCATE_MODEL", "phi3:mini")
DEFAULT_MAX_TOKENS = 160
MAX_RAW_TURNS = 3
UPSHOT_CHAR_LIMIT = 180


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("expert_model", DEFAULT_EXPERT_MODEL)
    st.session_state.setdefault("advocate_model", DEFAULT_ADVOCATE_MODEL)
    st.session_state.setdefault("advocate_persona", "")
    st.session_state.setdefault("route_to", "LLM Expert")
    st.session_state.setdefault("pending_forward_route", False)


def build_system_prompt(target: Literal["LLM Expert", "LLM Advocate"]) -> str:
    if target == "LLM Expert":
        return (
            "You are LLM Expert. Answer the user's question directly and concisely "
            "in 2-3 short sentences."
        )

    persona = st.session_state.get("advocate_persona", "").strip()
    if persona:
        return (
            f"You are {persona}. Speak strictly in the first person as {persona}. "
            "Do not repeat LLM Expert's wording; give your own perspective or pushback "
            "in 2-3 short sentences."
        )
    return (
        "You are LLM Advocate. Challenge assumptions or state one practical trade-off "
        "in 2-3 short sentences."
    )


def clean_reply(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(LLM\s+Advoc?a?te|LLM\s+Expert|User)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def build_condensed_payload(raw_messages: list[dict[str, str]]) -> str:
    if not raw_messages:
        return "Hello"

    older = raw_messages[:-MAX_RAW_TURNS] if len(raw_messages) > MAX_RAW_TURNS else []
    recent = raw_messages[-MAX_RAW_TURNS:]
    parts: list[str] = []

    if older:
        upshots = []
        for item in older:
            compact = " ".join(item["content"].split())
            if len(compact) > UPSHOT_CHAR_LIMIT:
                compact = compact[: UPSHOT_CHAR_LIMIT - 3].rstrip() + "..."
            upshots.append(f"- {item['speaker']}: {compact}")
        parts.append("Earlier upshot:\n" + "\n".join(upshots))

    recent_lines = [f"{msg['speaker']}: {msg['content']}" for msg in recent]
    parts.append("Recent turns:\n" + "\n".join(recent_lines))
    return "\n\n".join(parts)


def call_ollama(model: str, system_prompt: str, user_payload: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": DEFAULT_MAX_TOKENS,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def run_model_turn(
    target: Literal["LLM Expert", "LLM Advocate"],
    prompt_text: str | None = None,
) -> None:
    if prompt_text:
        st.session_state.messages.append(
            {"role": "user", "speaker": "User", "content": prompt_text}
        )

    model = (
        st.session_state.expert_model
        if target == "LLM Expert"
        else st.session_state.advocate_model
    )
    system_prompt = build_system_prompt(target)
    user_payload = build_condensed_payload(st.session_state.messages)

    try:
        raw_reply = call_ollama(
            model=model,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )
        reply = clean_reply(raw_reply)
    except Exception as exc:
        reply = f"[{model} error: {exc}]"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "speaker": target,
            "content": reply,
        }
    )


def trigger_forward_route(next_target: Literal["LLM Expert", "LLM Advocate"]) -> None:
    st.session_state.route_to = next_target
    st.session_state.pending_forward_route = True


def main() -> None:
    st.set_page_config(page_title="Agent Advocate", layout="centered")
    init_state()

    if st.session_state.pending_forward_route:
        st.session_state.pending_forward_route = False
        with st.spinner(f"Routing to {st.session_state.route_to}..."):
            run_model_turn(target=st.session_state.route_to, prompt_text=None)

    st.title("Agent Advocate")
    st.caption("Local two-model Streamlit workspace with explicit user routing.")

    with st.sidebar:
        st.subheader("Models & Persona")
        st.text_input("LLM Expert Model", key="expert_model")
        st.text_input("LLM Advocate Model", key="advocate_model")
        st.text_input(
            "Advocate Persona",
            key="advocate_persona",
            placeholder="e.g., a lifelong Republican farmer",
        )
        if st.button("Clear Thread"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**{msg['speaker']}**")
            st.write(msg["content"])

    with st.form("route_form", clear_on_submit=True):
        prompt_text = st.text_area(
            "Prompt",
            height=90,
            placeholder="Type a question or prompt...",
        )
        route_target: Literal["LLM Expert", "LLM Advocate"] = st.radio(
            "Route To",
            options=["LLM Expert", "LLM Advocate"],
            key="route_to",
            horizontal=True,
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        cleaned = prompt_text.strip()
        if cleaned:
            with st.spinner(f"Routing to {route_target}..."):
                run_model_turn(target=route_target, prompt_text=cleaned)
            st.rerun()

    if st.session_state.messages:
        st.caption("Route the latest turn directly to the other model:")
        other: Literal["LLM Expert", "LLM Advocate"] = (
            "LLM Advocate"
            if st.session_state.route_to == "LLM Expert"
            else "LLM Expert"
        )
        st.button(
            f"Route Last Turn to {other}",
            on_click=trigger_forward_route,
            args=(other,),
        )


if __name__ == "__main__":
    main()