import { useEffect } from "react";
import { createPortal } from "react-dom";
import { IconClose } from "./icons";

// Self-contained full-screen image preview. Rendered via a portal onto
// document.body so it escapes any transformed/overflow-clipped ancestor.
// Close on Esc, on backdrop click, or via the corner ✕ button. Body scroll
// is locked while open. No external deps.
export default function Lightbox({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt?: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  return createPortal(
    <div
      className="lightbox-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Image preview"
      onClick={onClose}
    >
      <button
        type="button"
        className="lightbox-close"
        onClick={onClose}
        aria-label="Close preview"
      >
        <IconClose />
      </button>
      {/* stopPropagation so clicking the image itself doesn't close */}
      <img
        className="lightbox-img"
        src={src}
        alt={alt || "Enlarged image"}
        onClick={(e) => e.stopPropagation()}
      />
    </div>,
    document.body,
  );
}
