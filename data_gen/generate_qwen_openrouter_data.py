import argparse
import concurrent.futures
import json
import os
import re
import time
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib import error, request


DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Answer the question and write the final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{prompt}\n\n"
    "Answer:"
)

SHORTER_ANSWERS_PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Provide a short explanation, and then write the final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{prompt}\n\n"
    "Answer:"
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b"
DEFAULT_API_KEY_ENVS = (
    "OPENROUTER_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENROUTER_KEY",
)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--math_json", required=True, help="Path to MATH train JSON.")
    parser.add_argument("--dataset_size", type=int, default=12000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out_file", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--retry_delay", type=float, default=3.0)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shorter_answers_prompt", action="store_true")
    parser.add_argument(
        "--site_url",
        default=os.environ.get("OPENROUTER_SITE_URL"),
        help="Optional HTTP-Referer header for OpenRouter rankings/analytics.",
    )
    parser.add_argument(
        "--app_name",
        default=os.environ.get("OPENROUTER_APP_NAME", "qwen3-nano-math-reasoner"),
        help="Optional X-Title header for OpenRouter rankings/analytics.",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help=(
            "Environment variable containing the OpenRouter API key. "
            "If unset, common OpenRouter aliases are also checked."
        ),
    )
    parser.add_argument(
        "--env_file",
        default=".env",
        help="Optional dotenv file to load before checking API key env vars.",
    )
    return parser.parse_args()


def render_prompt(prompt, shorter_answers_prompt=False):
    template = SHORTER_ANSWERS_PROMPT_TEMPLATE if shorter_answers_prompt else DEFAULT_PROMPT_TEMPLATE
    return template.format(prompt=prompt)


def model_to_filename(model_name, dataset_size):
    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return f"math_train_qwen_{safe_model}_{dataset_size // 1000}k.json"


def load_dotenv_file(path):
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_api_key(primary_env):
    env_names = []
    for name in (primary_env, *DEFAULT_API_KEY_ENVS):
        if name and name not in env_names:
            env_names.append(name)

    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value, name

    raise SystemExit(
        "OpenRouter API key missing. Set one of: "
        + ", ".join(env_names)
        + "."
    )


def content_to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(content_to_text(item) for item in value)
    if isinstance(value, dict):
        chunks = []
        for key in (
            "content",
            "text",
            "value",
            "output_text",
            "reasoning",
            "reasoning_content",
            "thinking",
            "message",
            "parts",
        ):
            if key in value:
                text = content_to_text(value[key])
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
        return "".join(
            content_to_text(nested)
            for key, nested in value.items()
            if key not in {"id", "type", "role", "index", "finish_reason", "logprobs", "usage"}
        )
    if value is None:
        return ""
    return str(value)


def split_think_tags(content):
    match = re.match(r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\Z", content, flags=re.DOTALL)
    if not match:
        return "", content
    return match.group(1).strip(), match.group(2).strip()


def parse_openrouter_response(decoded):
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter response missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("OpenRouter response has invalid choices format.")

    message = first_choice.get("message")
    content = ""
    thinking = ""
    if isinstance(message, dict):
        content = content_to_text(message.get("content"))
        thinking = (
            content_to_text(message.get("reasoning"))
            or content_to_text(message.get("reasoning_content"))
            or content_to_text(message.get("thinking"))
        )
    elif isinstance(message, str):
        content = content_to_text(message)

    if not content:
        content = content_to_text(first_choice.get("text")) or content_to_text(first_choice.get("delta"))

    if not thinking and content:
        thinking, stripped_content = split_think_tags(content)
        if thinking:
            content = stripped_content

    if not content and thinking:
        content = thinking

    if not content:
        raise RuntimeError(
            "OpenRouter response did not contain parseable assistant content. "
            f"choice_keys={sorted(first_choice.keys())}, root_keys={sorted(decoded.keys())}"
        )

    return {
        "message_thinking": thinking,
        "message_content": content,
        "usage": decoded.get("usage"),
        "finish_reason": first_choice.get("finish_reason"),
    }


def query_openrouter_chat(
    prompt,
    model,
    api_key,
    max_new_tokens,
    temperature,
    top_p,
    timeout,
    max_retries,
    retry_delay,
    site_url=None,
    app_name=None,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(OPENROUTER_CHAT_URL, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return parse_openrouter_response(json.loads(body))
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} from OpenRouter: {err_body}")
            if exc.code in {400, 401, 402, 403, 404}:
                break
        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            IncompleteRead,
            RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(retry_delay * (2 ** (attempt - 1)), 120.0))

    raise RuntimeError(
        f"Failed to query OpenRouter after {max_retries} attempt(s). Last error: {last_error}"
    )


def write_rows_json_incremental(rows, out_file):
    tmp_file = out_file.with_name(f"{out_file.name}.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_file.replace(out_file)


def load_resume_rows(out_file):
    with out_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Resume file must contain a JSON array. Got {type(rows).__name__}.")
    return rows


def validate_resume_rows(rows, selected_data):
    if len(rows) > len(selected_data):
        raise ValueError(f"Resume file has {len(rows)} rows, but dataset has {len(selected_data)}.")
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Resume row {idx} is not a JSON object.")
        if row.get("problem") != selected_data[idx - 1]["problem"]:
            raise ValueError(
                f"Resume row {idx} does not match the current dataset. "
                "Use a different output file or disable --resume."
            )


def generate_row(
    row,
    shorter_answers_prompt,
    model,
    api_key,
    max_new_tokens,
    temperature,
    top_p,
    timeout,
    max_retries,
    retry_delay,
    site_url,
    app_name,
):
    response = query_openrouter_chat(
        prompt=render_prompt(row["problem"], shorter_answers_prompt=shorter_answers_prompt),
        model=model,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        site_url=site_url,
        app_name=app_name,
    )
    answer = row.get("answer", row.get("gtruth_answer"))
    return {
        "problem": row["problem"],
        "gtruth_answer": answer,
        "answer": answer,
        "message_thinking": response["message_thinking"],
        "message_content": response["message_content"],
        "provider": "openrouter",
        "teacher_model": model,
        "usage": response["usage"],
        "finish_reason": response["finish_reason"],
    }


def progress_message(processed, total, start_time):
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta_seconds = remaining / rate if rate > 0 else 0.0
    return (
        f"{processed}/{total} | elapsed {elapsed / 60:.1f} min | "
        f"{rate * 60:.2f} rows/min | ETA {eta_seconds / 60:.1f} min"
    )


def main():
    args = parse_args()
    if args.num_processes < 1:
        raise SystemExit("--num_processes must be >= 1.")

    if args.env_file:
        load_dotenv_file(args.env_file)
    api_key, api_key_env = resolve_api_key(args.api_key_env)

    with Path(args.math_json).expanduser().open("r", encoding="utf-8") as f:
        math_data = json.load(f)
    if not isinstance(math_data, list):
        raise SystemExit("--math_json must contain a JSON list.")

    selected_data = math_data[: args.dataset_size]
    if len(selected_data) != args.dataset_size:
        raise SystemExit(f"Requested {args.dataset_size} rows but found {len(selected_data)}.")

    out_file = (
        Path(args.out_file).expanduser().resolve()
        if args.out_file
        else (Path.cwd() / model_to_filename(args.model, args.dataset_size)).resolve()
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start_idx = 0
    if args.resume and out_file.exists():
        rows = load_resume_rows(out_file)
        validate_resume_rows(rows, selected_data)
        start_idx = len(rows)
        print(f"Resume enabled: {start_idx}/{len(selected_data)} rows already completed.")
    else:
        if args.resume:
            print(f"Resume enabled but output file does not exist yet: {out_file}")
        write_rows_json_incremental(rows, out_file)

    if start_idx >= len(selected_data):
        print(f"All {len(selected_data)} rows are already completed: {out_file}")
        return

    print(f"Using OpenRouter API: {OPENROUTER_CHAT_URL}")
    print(f"Teacher model: {args.model}")
    print(f"API key env: {api_key_env}")
    query_openrouter_chat(
        prompt="Reply with OK.",
        model=args.model,
        api_key=api_key,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        site_url=args.site_url,
        app_name=args.app_name,
    )
    print("Model ready")

    remaining_data = selected_data[start_idx:]
    remaining_total = len(remaining_data)
    start_time = time.time()

    if args.num_processes == 1:
        for offset, row in enumerate(remaining_data, start=1):
            rows.append(
                generate_row(
                    row,
                    args.shorter_answers_prompt,
                    args.model,
                    api_key,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    args.timeout,
                    args.max_retries,
                    args.retry_delay,
                    args.site_url,
                    args.app_name,
                )
            )
            write_rows_json_incremental(rows, out_file)
            msg = progress_message(offset, remaining_total, start_time)
            if args.verbose:
                print(f"{start_idx + offset}/{len(selected_data)} -> {rows[-1]['message_content']}")
            print(msg, end="\r", flush=True)
    else:
        print(f"Parallel requests enabled: {args.num_processes}")
        next_submit = 0
        next_write = 0
        futures = {}
        completed_rows = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_processes) as executor:
            while next_write < remaining_total:
                while next_submit < remaining_total and len(futures) < args.num_processes:
                    future = executor.submit(
                        generate_row,
                        remaining_data[next_submit],
                        args.shorter_answers_prompt,
                        args.model,
                        api_key,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        args.timeout,
                        args.max_retries,
                        args.retry_delay,
                        args.site_url,
                        args.app_name,
                    )
                    futures[future] = next_submit
                    next_submit += 1

                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                failed_at = None
                failed_exc = None
                for future in done:
                    offset0 = futures.pop(future)
                    try:
                        completed_rows[offset0] = future.result()
                    except Exception as exc:
                        failed_at = offset0
                        failed_exc = exc
                        break

                while next_write in completed_rows:
                    rows.append(completed_rows.pop(next_write))
                    write_rows_json_incremental(rows, out_file)
                    processed = next_write + 1
                    print(progress_message(processed, remaining_total, start_time), end="\r", flush=True)
                    next_write += 1

                if failed_at is not None:
                    for pending_future in futures:
                        pending_future.cancel()
                    failing_idx = start_idx + failed_at + 1
                    raise RuntimeError(f"Generation failed at dataset row {failing_idx}.") from failed_exc

    write_rows_json_incremental(rows, out_file)
    seconds_elapsed = time.time() - start_time
    print(f"\nTotal time: {seconds_elapsed / 60:.1f} min")
    print(f"Wrote {len(rows)} rows to: {out_file}")


if __name__ == "__main__":
    main()
