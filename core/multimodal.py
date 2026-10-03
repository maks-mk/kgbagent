from __future__ import annotations

import base64
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtGui import QImage, QImageReader

from core.constants import BASE_DIR

IMAGE_CONTENT_BLOCK_TYPES = {"image", "image_url", "input_image"}
IMAGE_CONTENT_OMITTED_PLACEHOLDER = (
    "Previous image input omitted because the current model does not support image input."
)
IMAGE_INPUT_PROFILE_KEYS = ("image_input", "imageInputs", "image_inputs")
DEFAULT_MODEL_CAPABILITIES = {"image_input_supported": False}


def normalize_model_capabilities(profile: Any) -> dict[str, bool]:
    profile_payload = profile if isinstance(profile, dict) else {}
    if "image_input_supported" in profile_payload:
        return {"image_input_supported": bool(profile_payload.get("image_input_supported"))}
    image_input_supported = False
    for key in IMAGE_INPUT_PROFILE_KEYS:
        if key in profile_payload:
            image_input_supported = bool(profile_payload.get(key))
            break
    return {"image_input_supported": image_input_supported}


def extract_model_capabilities(llm: Any) -> dict[str, bool]:
    raw_profile = getattr(llm, "profile", None)
    if callable(raw_profile):
        try:
            raw_profile = raw_profile()
        except TypeError:
            pass
        except Exception:
            raw_profile = None
    return normalize_model_capabilities(raw_profile)


def profile_supports_image_input(profile: Any) -> bool | None:
    if not isinstance(profile, dict) or "supports_image_input" not in profile:
        return None
    return bool(profile.get("supports_image_input"))


def resolve_model_capabilities(profile: Any = None, runtime_capabilities: Any = None) -> dict[str, bool]:
    capabilities = normalize_model_capabilities(runtime_capabilities)
    manual_override = profile_supports_image_input(profile)
    if manual_override is not None:
        capabilities["image_input_supported"] = manual_override
    return capabilities


def normalize_image_attachments(attachments: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in attachments or []:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip()
        if not path:
            continue
        attachment = {
            "id": str(raw.get("id") or uuid.uuid4().hex).strip() or uuid.uuid4().hex,
            "session_id": str(raw.get("session_id") or "").strip(),
            "source": str(raw.get("source") or "file").strip() or "file",
            "path": path,
            "mime_type": str(raw.get("mime_type") or "").strip(),
            "file_name": str(raw.get("file_name") or Path(path).name).strip() or Path(path).name,
            "size_bytes": int(raw.get("size_bytes") or 0),
            "width": int(raw.get("width") or 0),
            "height": int(raw.get("height") or 0),
        }
        normalized.append(attachment)
    return normalized


def attachments_root(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else BASE_DIR
    return root / ".agent_state" / "attachments"


def session_attachments_dir(session_id: str, *, base_dir: str | Path | None = None) -> Path:
    directory = attachments_root(base_dir) / (str(session_id or "").strip() or "__draft__")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _image_dimensions(path: str | Path) -> tuple[int, int]:
    reader = QImageReader(str(path))
    size = reader.size()
    width = int(size.width()) if size.isValid() else 0
    height = int(size.height()) if size.isValid() else 0
    return width, height


def _image_mime_type(path: str | Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def can_read_image_file(path: str | Path) -> bool:
    reader = QImageReader(str(path))
    return reader.canRead()


def import_image_attachment_from_file(
    source_path: str | Path,
    *,
    session_id: str,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        raise ValueError("Image file does not exist.")
    if not can_read_image_file(source):
        raise ValueError("Unsupported image file.")

    attachment_id = uuid.uuid4().hex
    suffix = source.suffix or ".png"
    target_dir = session_attachments_dir(session_id, base_dir=base_dir)
    target_path = target_dir / f"img-{attachment_id}{suffix}"
    shutil.copy2(source, target_path)
    width, height = _image_dimensions(target_path)
    if width <= 0 or height <= 0:
        target_path.unlink(missing_ok=True)
        raise ValueError("Unable to read image dimensions.")
    return {
        "id": attachment_id,
        "session_id": str(session_id or "").strip(),
        "source": "file",
        "path": str(target_path.resolve()),
        "mime_type": _image_mime_type(target_path),
        "file_name": source.name,
        "size_bytes": int(target_path.stat().st_size),
        "width": width,
        "height": height,
    }


def import_image_attachment_from_qimage(
    image: Any,
    *,
    session_id: str,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    if hasattr(image, "toImage"):
        image = image.toImage()
    if not isinstance(image, QImage):
        raise ValueError("Clipboard image is empty.")
    if image.isNull():
        raise ValueError("Clipboard image is empty.")
    attachment_id = uuid.uuid4().hex
    target_dir = session_attachments_dir(session_id, base_dir=base_dir)
    target_path = target_dir / f"clip-{attachment_id}.png"
    if not image.save(str(target_path), "PNG"):
        raise ValueError("Failed to save clipboard image.")
    width, height = int(image.width()), int(image.height())
    return {
        "id": attachment_id,
        "session_id": str(session_id or "").strip(),
        "source": "clipboard",
        "path": str(target_path.resolve()),
        "mime_type": "image/png",
        "file_name": target_path.name,
        "size_bytes": int(target_path.stat().st_size),
        "width": width,
        "height": height,
    }


def build_user_message_content(user_text: str, attachments: Any) -> str | list[dict[str, Any]]:
    normalized_attachments = normalize_image_attachments(attachments)
    if not normalized_attachments:
        return str(user_text or "")

    parts: list[dict[str, Any]] = []
    text = str(user_text or "")
    if text.strip():
        parts.append({"type": "text", "text": text})
    for attachment in normalized_attachments:
        parts.append(
            {
                "type": "image",
                "path": str(attachment.get("path") or ""),
                "mime_type": str(attachment.get("mime_type") or _image_mime_type(attachment.get("path") or "")),
                "file_name": str(attachment.get("file_name") or ""),
                "attachment_id": str(attachment.get("id") or ""),
                "source": str(attachment.get("source") or "file"),
                "width": int(attachment.get("width") or 0),
                "height": int(attachment.get("height") or 0),
                "size_bytes": int(attachment.get("size_bytes") or 0),
            }
        )
    return parts


def extract_user_turn_data(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []

    attachments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for item in content if isinstance(content, list) else [content]:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "text" or "text" in item:
                text_parts.append(str(item.get("text") or ""))
                continue
            if item_type == "image":
                path = str(item.get("path") or "").strip()
                if path:
                    attachments.append(
                        {
                            "id": str(item.get("attachment_id") or item.get("id") or uuid.uuid4().hex).strip()
                            or uuid.uuid4().hex,
                            "session_id": "",
                            "source": str(item.get("source") or "file").strip() or "file",
                            "path": path,
                            "mime_type": str(item.get("mime_type") or _image_mime_type(path)).strip()
                            or _image_mime_type(path),
                            "file_name": str(item.get("file_name") or Path(path).name).strip() or Path(path).name,
                            "size_bytes": int(item.get("size_bytes") or 0),
                            "width": int(item.get("width") or 0),
                            "height": int(item.get("height") or 0),
                        }
                    )
                continue
            if item_type == "image_url":
                image_url_payload = item.get("image_url")
                image_url = ""
                if isinstance(image_url_payload, dict):
                    image_url = str(image_url_payload.get("url") or "").strip()
                elif isinstance(image_url_payload, str):
                    image_url = image_url_payload.strip()
                if image_url.startswith("data:"):
                    mime_type = "image/png"
                    header, _, data = image_url.partition(",")
                    if ";base64" in header:
                        mime_type = header[5:].split(";", 1)[0].strip() or mime_type
                    attachments.append(
                        {
                            "id": str(item.get("attachment_id") or item.get("id") or uuid.uuid4().hex).strip()
                            or uuid.uuid4().hex,
                            "session_id": "",
                            "source": str(item.get("source") or "file").strip() or "file",
                            "path": "",
                            "mime_type": mime_type,
                            "file_name": str(item.get("file_name") or "inline-image").strip() or "inline-image",
                            "size_bytes": len(data),
                            "width": int(item.get("width") or 0),
                            "height": int(item.get("height") or 0),
                        }
                    )
                continue
            nested_content = item.get("content")
            if nested_content is not None:
                nested_text, nested_attachments = extract_user_turn_data(nested_content)
                if nested_text:
                    text_parts.append(nested_text)
                attachments.extend(nested_attachments)
                continue

    return "".join(text_parts), normalize_image_attachments(attachments)


def human_message_has_image_content(content: Any) -> bool:
    if isinstance(content, str):
        return False
    for item in content if isinstance(content, list) else [content]:
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in IMAGE_CONTENT_BLOCK_TYPES or (not item_type and "image_url" in item):
                return True
            nested_content = item.get("content")
            if nested_content is not None and human_message_has_image_content(nested_content):
                return True
    return False


def strip_image_content_from_message_content(content: Any) -> tuple[Any, int]:
    if isinstance(content, str):
        return content, 0

    items = content if isinstance(content, list) else [content]
    sanitized_items: list[Any] = []
    removed_blocks = 0

    for item in items:
        if isinstance(item, str):
            sanitized_items.append(item)
            continue

        if not isinstance(item, dict):
            sanitized_items.append(item)
            continue

        item_type = str(item.get("type") or "").strip().lower()
        if item_type in IMAGE_CONTENT_BLOCK_TYPES or (not item_type and "image_url" in item):
            removed_blocks += 1
            continue

        nested_content = item.get("content")
        if nested_content is None:
            sanitized_items.append(item)
            continue

        sanitized_nested_content, nested_removed_blocks = strip_image_content_from_message_content(nested_content)
        if nested_removed_blocks:
            cloned_item = dict(item)
            cloned_item["content"] = sanitized_nested_content
            sanitized_items.append(cloned_item)
            removed_blocks += nested_removed_blocks
            continue

        sanitized_items.append(item)

    if removed_blocks == 0:
        return content, 0

    if not sanitized_items:
        return IMAGE_CONTENT_OMITTED_PLACEHOLDER, removed_blocks

    if isinstance(content, list):
        return sanitized_items, removed_blocks

    if len(sanitized_items) == 1:
        return sanitized_items[0], removed_blocks
    return sanitized_items, removed_blocks


def _as_anthropic_image_block(base64_data: str, mime_type: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_data,
        },
    }


def _as_openai_image_url_block(base64_data: str, mime_type: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_data}",
        },
    }


def materialize_user_message_content_for_model(content: Any, *, provider: str = "") -> Any:
    if not isinstance(content, list):
        return content

    provider_name = str(provider or "").strip().lower()
    materialized: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            materialized.append(item)
            continue

        item_type = str(item.get("type") or "").strip().lower()
        if item_type != "image":
            materialized.append(item)
            continue

        path = str(item.get("path") or "").strip()
        if not path:
            materialized.append(item)
            continue

        file_path = Path(path)
        if not file_path.exists():
            materialized.append(item)
            continue

        base64_data = base64.b64encode(file_path.read_bytes()).decode("ascii")
        mime_type = str(item.get("mime_type") or _image_mime_type(file_path))
        if provider_name == "openai":
            materialized.append(_as_openai_image_url_block(base64_data, mime_type))
            continue
        if provider_name == "anthropic":
            materialized.append(_as_anthropic_image_block(base64_data, mime_type))
            continue
        materialized.append(
            {
                "type": "image",
                "base64": base64_data,
                "mime_type": mime_type,
            }
        )

    return materialized


def normalize_request_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        text = str(payload.get("text") or "")
        attachments = normalize_image_attachments(payload.get("attachments"))
        chat_only = payload.get("chat_only") is True
    else:
        text = str(payload or "")
        attachments = []
        chat_only = False
    return {"text": text, "attachments": attachments, "chat_only": chat_only}


def request_has_content(payload: Any) -> bool:
    normalized = normalize_request_payload(payload)
    return bool(normalized["text"].strip() or normalized["attachments"])


def request_task_text(payload: Any) -> str:
    normalized = normalize_request_payload(payload)
    text = normalized["text"].strip()
    if text:
        return text
    if normalized["attachments"]:
        return "Analyze the attached images."
    return ""


def request_user_text(payload: Any) -> str:
    return normalize_request_payload(payload)["text"]
