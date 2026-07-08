from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Mapping

PAR_END = "<par_end>"
MAX_PARAMETER_BYTES = 40 * 1024
_BLOCK_RE = re.compile(r"^\s*<([^>]+)>\s*(?:[#;].*)?$")
_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_+\-./]+)(\s*=\s*)(.*?)(\s*(?:[#;].*)?)$")


@dataclass
class ParameterBlock:
    text: str
    values: dict[str, dict[str, str]]

    def get(self, block: str, key: str, default: str | None = None) -> str | None:
        return self.values.get(block, {}).get(key, default)

    def get_int(self, block: str, key: str, default: int | None = None) -> int | None:
        value = self.get(block, key)
        return default if value is None else int(value)

    def get_float(self, block: str, key: str, default: float | None = None) -> float | None:
        value = self.get(block, key)
        return default if value is None else float(value)

    def get_bool(self, block: str, key: str, default: bool | None = None) -> bool | None:
        value = self.get(block, key)
        if value is None:
            return default
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes"}:
            return True
        if lowered in {"false", "f", "0", "no"}:
            return False
        raise ValueError(f"invalid boolean <{block}>/{key}: {value!r}")

    def patched(self, updates: Mapping[tuple[str, str], object]) -> str:
        remaining = {(block, key): str(value) for (block, key), value in updates.items()}
        output: list[str] = []
        current: str | None = None
        for raw_line in self.text.splitlines(keepends=True):
            line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            newline = "\n" if raw_line.endswith("\n") else ""
            block_match = _BLOCK_RE.match(line)
            if block_match:
                current = block_match.group(1)
                output.append(raw_line)
                continue
            key_match = _KEY_RE.match(line)
            if current is not None and key_match:
                key = key_match.group(2)
                update_key = (current, key)
                if update_key in remaining:
                    prefix, key_text, sep, _old_value, suffix = key_match.groups()
                    output.append(f"{prefix}{key_text}{sep}{remaining.pop(update_key)}{suffix}{newline}")
                    continue
            if line.strip() == PAR_END:
                _append_missing_updates(output, current, remaining)
                remaining.clear()
                output.append(raw_line)
                continue
            output.append(raw_line)
        if remaining:
            raise ValueError("parameter block is missing <par_end>")
        return "".join(output)


def _append_missing_updates(
    output: list[str],
    current: str | None,
    remaining: dict[tuple[str, str], str],
) -> None:
    if not remaining:
        return
    by_block: dict[str, list[tuple[str, str]]] = {}
    for (block, key), value in remaining.items():
        by_block.setdefault(block, []).append((key, value))
    if output and output[-1] and not output[-1].endswith("\n"):
        output[-1] += "\n"
    for block, entries in by_block.items():
        if current != block:
            output.append(f"<{block}>\n")
        for key, value in sorted(entries):
            output.append(f"{key} = {value}\n")


def parse_parameter_text(text: str) -> ParameterBlock:
    values: dict[str, dict[str, str]] = {}
    current: str | None = None
    saw_end = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == PAR_END:
            saw_end = True
            break
        block_match = _BLOCK_RE.match(raw_line)
        if block_match:
            current = block_match.group(1)
            values.setdefault(current, {})
            continue
        key_match = _KEY_RE.match(raw_line)
        if current is not None and key_match:
            key = key_match.group(2)
            value = key_match.group(4).strip()
            values[current][key] = value
    if not saw_end:
        raise ValueError("parameter block does not contain <par_end>")
    return ParameterBlock(text=text, values=values)


def read_parameter_block(stream: BinaryIO) -> ParameterBlock:
    chunks: list[bytes] = []
    total = 0
    while total <= MAX_PARAMETER_BYTES:
        line = stream.readline()
        if not line:
            raise EOFError("restart parameter block ended before <par_end>")
        chunks.append(line)
        total += len(line)
        if line.strip() == PAR_END.encode("ascii"):
            try:
                return parse_parameter_text(b"".join(chunks).decode("ascii"))
            except UnicodeDecodeError as exc:
                raise ValueError("parameter block is not ASCII") from exc
    raise ValueError(f"parameter block exceeds {MAX_PARAMETER_BYTES} bytes before <par_end>")
