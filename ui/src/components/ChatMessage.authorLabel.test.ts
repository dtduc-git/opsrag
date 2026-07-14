import { describe, it, expect } from "vitest";
import { displayAuthorLabel, type Message, type ChatMessageContext } from "./ChatMessage";

function userMsg(authorEmail: string | null, authorName: string | null): Message {
  return { role: "user", content: "", authorEmail, authorName };
}

function ctx(viewerEmail: string | null): ChatMessageContext {
  return { viewerEmail };
}

describe("displayAuthorLabel", () => {
  it("case-insensitive email match against the viewer → You", () => {
    expect(displayAuthorLabel(userMsg("a@x", null), ctx("A@X"))).toBe("You");
  });

  it("different author with a name → the teammate's name", () => {
    expect(displayAuthorLabel(userMsg("a@x", "Ann"), ctx("b@x"))).toBe("Ann");
  });

  it("different author, no name → email local-part fallback", () => {
    expect(displayAuthorLabel(userMsg("a@x", null), ctx("b@x"))).toBe("a");
  });

  it("THE F1 FIX: null authorEmail but a resolved authorName → the name, not You", () => {
    expect(displayAuthorLabel(userMsg(null, "Ann"), ctx("b@x"))).toBe("Ann");
  });

  it("null authorEmail and null authorName, viewer present → unattributable → You", () => {
    expect(displayAuthorLabel(userMsg(null, null), ctx("b@x"))).toBe("You");
  });

  it("null authorEmail, null authorName, null viewer → legacy/anon → You", () => {
    expect(displayAuthorLabel(userMsg(null, null), ctx(null))).toBe("You");
  });

  it("assistant role → always OpsRAG regardless of author fields", () => {
    const msg: Message = { role: "assistant", content: "", authorEmail: "a@x", authorName: "Ann" };
    expect(displayAuthorLabel(msg, ctx("a@x"))).toBe("OpsRAG");
  });
});
