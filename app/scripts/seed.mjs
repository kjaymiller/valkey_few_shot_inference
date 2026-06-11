#!/usr/bin/env node
// Seed the Valkey example bank by POSTing past "successful" inferences to the
// running Spin app's /feedback endpoint. The app handles embedding + indexing,
// so this script needs no Valkey client of its own.
//
// Usage:
//   spin up --variable valkey_url="$VALKEY_URL"   # in another terminal
//   APP_URL=http://127.0.0.1:3000 node scripts/seed.mjs

const APP_URL = process.env.APP_URL ?? "http://127.0.0.1:3000";

const EXAMPLES = [
  {
    input: "Write a friendly out-of-office reply for a 2-week vacation.",
    completion:
      "Thanks for your email! I'm out of office until the 24th with limited access to messages. For anything urgent, please reach out to my colleague Sam at sam@example.com. I'll get back to you as soon as I return.",
  },
  {
    input: "Summarize a quarterly sales report in one upbeat sentence.",
    completion:
      "Q3 was a standout quarter: revenue climbed 18% on strong demand in the enterprise segment, putting us comfortably ahead of plan heading into year-end.",
  },
  {
    input: "Draft a polite reminder for an overdue invoice.",
    completion:
      "Hi there — just a quick note that invoice #1042 was due last week and is still showing as unpaid. Could you let me know if there's anything you need from us to process it? Happy to help. Thank you!",
  },
  {
    input: "Explain what a vector database does to a non-technical manager.",
    completion:
      "A vector database stores information as lists of numbers that capture meaning, so instead of matching exact words it can find things that are similar in intent — like finding past support tickets that 'feel like' a new one, even if they use different wording.",
  },
  {
    input: "Write a one-line commit message for fixing a login redirect bug.",
    completion: "fix(auth): redirect to intended page after login instead of dashboard",
  },
  {
    input: "Give an encouraging response to a teammate whose deploy failed.",
    completion:
      "Deploys fail on everyone — the important part is you caught it before users did. Want to pair on the rollback and figure out the root cause together?",
  },
];

async function main() {
  console.log(`Seeding ${EXAMPLES.length} examples to ${APP_URL}/feedback ...`);
  for (const ex of EXAMPLES) {
    const res = await fetch(`${APP_URL}/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(ex),
    });
    const text = await res.text();
    if (!res.ok) {
      console.error(`  ✗ ${ex.input.slice(0, 40)}… -> ${res.status} ${text}`);
      process.exitCode = 1;
    } else {
      console.log(`  ✓ ${ex.input.slice(0, 40)}… -> ${text.trim()}`);
    }
  }
  console.log("Done. Try: curl -s $APP_URL/infer -d '{\"input\":\"Write an out-of-office note for a conference trip\"}'");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
