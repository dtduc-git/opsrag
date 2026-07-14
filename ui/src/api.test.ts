import { describe, it, expect } from "vitest";
import { slackPermalink, normalizeTitle } from "./api";

describe("slackPermalink", () => {
  it("builds a permalink from channel + dotted ts", () => {
    expect(slackPermalink("slack-thread:C08:1783938089.103429", "https://ws.slack.com")).toBe(
      "https://ws.slack.com/archives/C08/p1783938089103429",
    );
  });

  it("strips a trailing slash from the workspace url", () => {
    expect(slackPermalink("slack-thread:C08:1783938089.103429", "https://ws.slack.com/")).toBe(
      "https://ws.slack.com/archives/C08/p1783938089103429",
    );
  });

  it("returns null when workspace is null/empty/undefined", () => {
    const id = "slack-thread:C08:1783938089.103429";
    expect(slackPermalink(id, null)).toBeNull();
    expect(slackPermalink(id, "")).toBeNull();
    expect(slackPermalink(id, undefined)).toBeNull();
  });

  it("returns null for a slack-dm id (not slack-thread)", () => {
    expect(slackPermalink("slack-dm:C08", "https://ws.slack.com")).toBeNull();
  });

  it("returns null for a web (uuid) session id", () => {
    expect(slackPermalink("1a2b_uuid", "https://ws.slack.com")).toBeNull();
  });

  it("returns null for a non-slack platform thread id", () => {
    expect(slackPermalink("discord-thread:C1:123.456", "https://ws.slack.com")).toBeNull();
  });

  it("returns null when ts is the no-ts sentinel", () => {
    expect(slackPermalink("slack-thread:C08:no-ts", "https://ws.slack.com")).toBeNull();
  });

  it("returns null when channel is empty (sep<=0)", () => {
    expect(slackPermalink("slack-thread::1783938089.103429", "https://ws.slack.com")).toBeNull();
  });

  it("returns null when there is no second colon (indexOf -1)", () => {
    expect(slackPermalink("slack-thread:C08", "https://ws.slack.com")).toBeNull();
  });

  it("is valid (no-op replace) when ts has no dot", () => {
    expect(slackPermalink("slack-thread:C08:1783938089", "https://ws.slack.com")).toBe(
      "https://ws.slack.com/archives/C08/p1783938089",
    );
  });
});

describe("normalizeTitle", () => {
  it("returns empty string for null/undefined/empty", () => {
    expect(normalizeTitle(null)).toBe("");
    expect(normalizeTitle(undefined)).toBe("");
    expect(normalizeTitle("")).toBe("");
  });

  it("takes the first line and trims it", () => {
    expect(normalizeTitle("  hi \n more")).toBe("hi");
  });

  it("collapses internal multi-space runs", () => {
    expect(normalizeTitle("a   b")).toBe("a b");
  });

  it("truncates a 61-char string to 59 chars + ellipsis (len 60)", () => {
    const raw = "a".repeat(61);
    const out = normalizeTitle(raw);
    expect(out.length).toBe(60);
    expect(out).toBe("a".repeat(59) + "…");
  });

  it("leaves strings <= 60 chars unchanged", () => {
    const raw = "a".repeat(60);
    expect(normalizeTitle(raw)).toBe(raw);
  });
});
