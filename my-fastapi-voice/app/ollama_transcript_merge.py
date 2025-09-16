#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import requests

# ===== Prompt template tuned for your task =====
DEFAULT_TEMPLATE = """SYSTEM / INSTRUCTIONS (English & Persian):

Goal: Produce a single unified transcript (mixed Persian–English).
- Keep ALL medical terms in ENGLISH (Latin script): drug names, diseases, procedures, anatomy, lab markers, syndromes, genes, imaging terms, units (mg, mL, bpm), and abbreviations (e.g., MI, AF, ALT, AST, INR, CT, MRI).
- Do NOT add new information. Do NOT invent, hallucinate, or generalize beyond the provided documents. Only merge and lightly normalize grammar/punctuation.
- Preserve all facts. If two sources conflict and the conflict cannot be reconciled without adding information, do inference based on the context and choose the best one.
- Maintain the same overall informational density. Target total length between {min_chars} and {max_chars} characters (aim ≈ {target_chars} characters).
- Write primarily in Persian for general narrative text, but ALWAYS keep medical terms in English (Latin script) inside the Persian sentences.
- Preserve numeric values, dates, doses, and units exactly; do not round or convert.
- Keep paragraphing clean and readable. No headings, no metadata, no markdown fences in the final output.
- Output ONLY the final merged transcript text. No preface, no explanation.

هدف: یک متن واحد (ترکیبی فارسی–انگلیسی) تولید کن.
- تمام اصطلاحات پزشکی را به انگلیسی با حروف لاتین نگه دار (نام داروها، بیماری‌ها، پروسیجرها، اندام‌ها، آزمایش‌ها، اختصارات و واحدها).
- هیچ اطلاعات جدیدی اضافه نکن. فقط ترکیب/بازنویسی محدود برای روانی متن.
- اگر منابع اختلاف دارند و حل اختلاف باعث افزودن اطلاعات جدید می‌شود، خودت استنتاج کن و با توجه به من بهترین نتیجه را گزارش کن.
- طول متن خروجی تقریباً برابر با مجموع متون ورودی باشد (بین {min_chars} تا {max_chars} کاراکتر؛ هدف ≈ {target_chars}).
- اعداد/دوز/واحدها را دقیق و بدون تغییر نگه دار.
- فقط متن نهایی را خروجی بده؛ بدون مقدمه یا توضیح.

You are given {n_docs} transcript document(s) below. Combine them under the above rules:

{documents}
"""

def read_text_file(path: Path, max_chars: int = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n...\n[TRUNCATED]\n...\n" + tail
    return text

def build_documents_block(files: List[Path], max_chars: int = None) -> str:
    blocks = []
    for idx, fp in enumerate(files, start=1):
        content = read_text_file(fp, max_chars=max_chars)
        blocks.append(
            f"### Document {idx}: {fp.name}\n"
            f"```\n{content}\n```\n"
        )
    return "\n".join(blocks)

def compute_length_targets(files: List[Path], max_chars_per_doc: int = None, tol: float = 0.10):
    total = 0
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            raw = ""
        if max_chars_per_doc and max_chars_per_doc > 0 and len(raw) > max_chars_per_doc:
            total += max_chars_per_doc
        else:
            total += len(raw)
    target = total
    low = int(total * (1.0 - tol))
    high = int(total * (1.0 + tol))
    return target, low, high

def build_prompt(files: List[Path], question: str, template: str, max_chars: int, tol: float):
    target_chars, min_chars, max_chars_total = compute_length_targets(files, max_chars_per_doc=max_chars, tol=tol)
    docs_block = build_documents_block(files, max_chars=max_chars)
    prompt = template.format(
        question=question if question else "(no explicit question provided)",
        n_docs=len(files),
        documents=docs_block,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars_total,
    )
    return prompt

def call_ollama_chat_or_generate(
    host: str,
    model: str,
    prompt: str,
    out_path: Path,
    stream: bool = True,
    options: dict = None,
    keep_alive: str = None,
    timeout: int = 1200,
    use_chat: bool = False,
):
    """
    Sends the request to Ollama and writes output to both stdout and out_path.
    """
    base = host.rstrip("/")
    headers = {"Content-Type": "application/json"}

    if use_chat:
        url = base + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Follow the given instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
        }
    else:
        url = base + "/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": stream}

    if options:
        payload["options"] = options
    if keep_alive:
        payload["keep_alive"] = keep_alive

    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        if stream:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                final_obj = {}
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # fall back to raw
                        print(line, end="", flush=True)
                        fout.write(line)
                        fout.flush()
                        continue

                    token = None
                    if "message" in obj and "content" in obj["message"]:
                        token = obj["message"]["content"]
                    elif "response" in obj:
                        token = obj["response"]

                    if token:
                        print(token, end="", flush=True)
                        fout.write(token)
                        fout.flush()

                    if obj.get("done"):
                        final_obj = obj
                        if token is None or not str(token).endswith("\n"):
                            print()
                            fout.write("\n")
                        fout.flush()
                        return final_obj
        else:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            obj = r.json()
            text = obj.get("message", {}).get("content") or obj.get("response", "") or ""
            print(text, end="")
            fout.write(text)
            if not text.endswith("\n"):
                print()
                fout.write("\n")
            fout.flush()
            return obj

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge English/Persian transcripts into one mixed text (medical terms kept in English) and send to an Ollama model."
    )
    p.add_argument("files", nargs="+", help="Paths to transcript .txt files.")
    p.add_argument("-m", "--model", required=True, help="Model name (e.g., meditron:70b-q4_0).")
    p.add_argument("-q", "--question", default="", help="Optional extra instruction to include.")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                   help="Ollama base URL (default: http://127.0.0.1:11434 or $OLLAMA_HOST).")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help="Custom template (advanced).")
    p.add_argument("--use-chat", action="store_true", help="Use /api/chat instead of /api/generate.")
    p.add_argument("--show-prompt", action="store_true", help="Print the composed prompt before sending.")
    p.add_argument("--output", "-o", default="merged_output.txt",
                   help="Write the model output here (default: merged_output.txt).")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming output (prints on completion).")
    p.add_argument("--keep-alive", default=None, help="Keep the model loaded (e.g. '30m').")
    p.add_argument("--timeout", type=int, default=1200, help="HTTP timeout seconds (default 1200).")

    # Safety knobs
    p.add_argument("--max-chars-per-doc", type=int, default=None,
                   help="Soft cap per doc (characters) to avoid huge prompts.")
    p.add_argument("--length-tolerance", type=float, default=0.10,
                   help="± tolerance for target output length (default 0.10 = ±10%).")

    # Model options passed to Ollama
    p.add_argument("--num-ctx", type=int, default=4096, help="Context window tokens (default 4096).")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default 0.2).")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mirostat", type=int, default=None)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    return p.parse_args()

def collect_options_from_args(args) -> dict:
    opts = {}
    if args.num_ctx is not None:
        opts["num_ctx"] = args.num_ctx
    if args.temperature is not None:
        opts["temperature"] = args.temperature
    if args.top_k is not None:
        opts["top_k"] = args.top_k
    if args.top_p is not None:
        opts["top_p"] = args.top_p
    if args.seed is not None:
        opts["seed"] = args.seed
    if args.mirostat is not None:
        opts["mirostat"] = args.mirostat
    if args.repeat_penalty is not None:
        opts["repeat_penalty"] = args.repeat_penalty
    opts.setdefault("num_predict", -1)  # unlimited
    return opts

def main():
    args = parse_args()
    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            sys.stderr.write(f"[error] file not found: {f}\n")
            sys.exit(1)

    prompt = build_prompt(
        files=files,
        question=args.question,
        template=args.template,
        max_chars=args.max_chars_per_doc,
        tol=args.length_tolerance,
    )

    if args.show_prompt:
        print("===== PROMPT START =====")
        print(prompt)
        print("===== PROMPT END =====")

    options = collect_options_from_args(args)
    out_path = Path(args.output)

    try:
        result = call_ollama_chat_or_generate(
            host=args.host,
            model=args.model,
            prompt=prompt,
            out_path=out_path,
            stream=not args.no_stream,
            options=options if options else None,
            keep_alive=args.keep_alive,
            timeout=args.timeout,
            use_chat=args.use_chat,
        )
        print(f"[info] saved output to: {out_path}")
    except requests.RequestException as e:
        sys.stderr.write(f"\n[error] HTTP: {e}\n")
        sys.exit(2)

if __name__ == "__main__":
    main()


# python3 ollama_transcript_merge.py -m "alibayram/medgemma:27b" --show-prompt --use-chat /home/user01/my-fastapi-voice/app/outputs/1/texts/*.txt