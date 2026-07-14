import { describe, it, expect } from "vitest";
import { prettifySlackTokens } from "./slackText";

describe("prettifySlackTokens", () => {
  it("1. converts a bare user mention", () => {
    expect(prettifySlackTokens("<@U12345>")).toBe("@U12345");
  });

  it("2. converts multiple user mentions", () => {
    expect(prettifySlackTokens("hey <@U1> and <@W2>")).toBe("hey @U1 and @W2");
  });

  it("3. converts subteam mentions (labeled + unlabeled)", () => {
    expect(prettifySlackTokens("<!subteam^S1|oncall>")).toBe("@oncall");
    expect(prettifySlackTokens("<!subteam^S1>")).toBe("@oncall");
  });

  it("4. converts special mentions (@here / @channel)", () => {
    expect(prettifySlackTokens("<!here>")).toBe("@here");
    expect(prettifySlackTokens("<!channel>")).toBe("@channel");
  });

  it("5. converts a piped link to markdown link syntax", () => {
    expect(prettifySlackTokens("<https://x.io/a|the runbook>")).toBe(
      "[the runbook](https://x.io/a)",
    );
  });

  // 6. PLAN DISCREPANCY (NEEDS_CONTEXT — flagged, not silently resolved):
  // §3.2 row 6 of docs/superpowers/plans/2026-07-13-opsrag-webui-phase2.md
  // claims a standalone `<https://x.io/a>` (no pipe) degrades to
  // `https://x.io/a`. But the exact §2.2 HAS_SLACK_TOKEN guard is:
  //   /<[@#!]|<https?:\/\/[^>\s]+\|/
  // The second alternative REQUIRES a literal "|" in the token. A bare
  // `<url>` with no pipe and no other real Slack token in the string never
  // trips the guard, so prettifySlackTokens short-circuits and returns the
  // input untouched — verified via node before writing this spec. This test
  // asserts the ACTUAL behavior of the exactly-transcribed §2.2 code (the
  // guard regex is called out as load-bearing / must mirror the plan), not
  // the table's claimed output. See task report for the flagged row.
  it("6. [PLAN DISCREPANCY] a standalone unpiped link does NOT trip the guard, so it is left untouched (see comment above)", () => {
    expect(prettifySlackTokens("<https://x.io/a>")).toBe("<https://x.io/a>");
  });
  // A piped link — or any bare link co-occurring with a real Slack token —
  // does convert (RE_LINK handles both forms once the gate is open):
  it("6b. an unpiped link DOES convert once a real Slack token opens the gate", () => {
    expect(prettifySlackTokens("<@U1> see <https://x.io/a>")).toBe("@U1 see https://x.io/a");
  });

  it("7. converts channel mentions (labeled + unlabeled)", () => {
    expect(prettifySlackTokens("<#C1|general>")).toBe("#general");
    expect(prettifySlackTokens("<#C1>")).toBe("#channel");
  });

  it("8. bolds *word* only once a real Slack token is present elsewhere", () => {
    expect(prettifySlackTokens("deploy is *broken* now <@U1>")).toBe(
      "deploy is **broken** now @U1",
    );
  });

  it("9. plain text with only an emoji shortcode (no <…> token) is left untouched", () => {
    expect(prettifySlackTokens("all good :tada:")).toBe("all good :tada:");
  });

  it("10. GUARD: clean markdown with no Slack token is untouched, even with single-* emphasis", () => {
    expect(prettifySlackTokens("use *italic* and **bold**")).toBe("use *italic* and **bold**");
  });

  it("11. code spans are protected: inline code stays literal, fenced block stays raw, only prose converts", () => {
    expect(prettifySlackTokens("inline `<@U1>` stays literal, but <@U2> converts")).toBe(
      "inline `<@U1>` stays literal, but @U2 converts",
    );
    expect(prettifySlackTokens("```\n<@U1>\n```\nprose <@U2>")).toBe(
      "```\n<@U1>\n```\nprose @U2",
    );
  });

  it("12. combined tokens + emoji + bold in one message", () => {
    expect(
      prettifySlackTokens(
        "<@U1> please check <https://wiki/x|here> :eyes: — it is *urgent*",
      ),
    ).toBe("@U1 please check [here](https://wiki/x) :eyes: — it is **urgent**");
  });

  it("13. is idempotent: re-running on already-degraded output is a no-op", () => {
    const once = prettifySlackTokens(
      "<@U1> please check <https://wiki/x|here> :eyes: — it is *urgent*",
    );
    const twice = prettifySlackTokens(once);
    expect(twice).toBe(once);
  });

  it("14. leaves unrelated angle-bracket HTML-looking text alone, with and without a real token", () => {
    expect(prettifySlackTokens("<div>foo</div>")).toBe("<div>foo</div>");
    expect(prettifySlackTokens("<div>foo</div> <@U1>")).toBe("<div>foo</div> @U1");
  });

  it("15. empty/no-op inputs", () => {
    expect(prettifySlackTokens("")).toBe("");
    expect(prettifySlackTokens("plain text")).toBe("plain text");
  });

  it("16. RE_BOLD guard: *x*y is left unchanged (not over-bolded) even with a token elsewhere", () => {
    expect(prettifySlackTokens("*x*y <@U1>")).toBe("*x*y @U1");
  });
});
