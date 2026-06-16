import { useState, useRef, useEffect, type KeyboardEvent, type ClipboardEvent, type DragEvent, type ChangeEvent } from "react";
import { IconSend, IconClose } from "./icons";

// A pending (not-yet-sent) image attachment held in the composer.
//   - `dataUrl` drives the thumbnail + the optimistic transcript render
//   - `b64` (base64, no data: prefix) is what we send to the backend
export type PendingImage = { mime: string; dataUrl: string; b64: string; name: string };

// Mirror the server-side cap (max 4 images) for snappier UX; the backend
// still enforces it authoritatively (count / size / mime → HTTP 400).
const MAX_IMAGES = 4;

function readFileAsImage(file: File): Promise<PendingImage> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const dataUrl = String(r.result);
      const [, mime = "image/png", b64 = ""] =
        /^data:(.*?);base64,(.*)$/.exec(dataUrl) || [];
      resolve({ mime, dataUrl, b64, name: file.name });
    };
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

interface Props {
  onSend: (payload: { text: string; images: PendingImage[] }) => void;
  disabled: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const [images, setImages] = useState<PendingImage[]>([]);
  const [dragging, setDragging] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Focus on mount + when becoming enabled.
  useEffect(() => { if (!disabled) ref.current?.focus(); }, [disabled]);

  // ⌘K / Ctrl+K to focus input from anywhere.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        ref.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Auto-grow textarea height.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  // Filter to images, decode each, and append up to the client-side cap.
  const addFiles = async (files: FileList | File[]) => {
    const picked = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!picked.length) return;
    const decoded = await Promise.all(picked.map(readFileAsImage));
    setImages((prev) => [...prev, ...decoded].slice(0, MAX_IMAGES));
  };

  const removeImage = (idx: number) =>
    setImages((prev) => prev.filter((_, i) => i !== idx));

  const onPickFiles = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) void addFiles(e.target.files);
    // Reset so picking the same file again still fires onChange.
    e.target.value = "";
  };

  const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = e.clipboardData?.files;
    if (files && files.length && Array.from(files).some((f) => f.type.startsWith("image/"))) {
      e.preventDefault();
      void addFiles(files);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer?.files?.length) void addFiles(e.dataTransfer.files);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!dragging) setDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
  };

  const handleSubmit = () => {
    const q = value.trim();
    // Allow sending images with no text, but require at least one of them.
    if ((!q && images.length === 0) || disabled) return;
    onSend({ text: q, images });
    setValue("");
    setImages([]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const canSend = !disabled && (Boolean(value.trim()) || images.length > 0);

  return (
    <div className="input-area">
      {images.length > 0 && (
        <div className="input-thumbs">
          {images.map((img, i) => (
            <div className="input-thumb" key={`${img.name}-${i}`}>
              <img src={img.dataUrl} alt={img.name || `attachment ${i + 1}`} />
              <button
                type="button"
                className="input-thumb-remove"
                onClick={() => removeImage(i)}
                aria-label={`Remove ${img.name || "image"}`}
              >
                <IconClose />
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        className={`input-shell${dragging ? " dragging" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={onPickFiles}
        />
        <button
          type="button"
          className="attach-btn"
          onClick={() => fileRef.current?.click()}
          disabled={disabled || images.length >= MAX_IMAGES}
          aria-label="Attach image"
          title={images.length >= MAX_IMAGES ? `Up to ${MAX_IMAGES} images` : "Attach image"}
        >
          📎
        </button>
        <textarea
          ref={ref}
          rows={1}
          className="input-field"
          placeholder="Ask about your infra — services, alerts, runbooks, postmortems..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          disabled={disabled}
        />
        <button
          className="send-btn"
          onClick={handleSubmit}
          disabled={!canSend}
          aria-label="Send"
        >
          <IconSend />
        </button>
      </div>
      <div className="input-hint">
        <span><kbd>Enter</kbd> send</span>
        <span><kbd>Shift</kbd>+<kbd>Enter</kbd> new line</span>
        <span><kbd>⌘</kbd>+<kbd>K</kbd> focus</span>
        <span>📎 / paste / drop image</span>
      </div>
    </div>
  );
}
