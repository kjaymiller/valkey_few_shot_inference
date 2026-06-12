#!/usr/bin/env node
// Seed the Valkey example bank by POSTing past "successful" inferences to the
// running Spin app's /feedback endpoint. The app handles embedding + indexing,
// so this script needs no Valkey client of its own.
//
// Usage:
//   spin up --variable valkey_url="$VALKEY_URL"   # in another terminal
//   APP_URL=http://127.0.0.1:3000 node scripts/seed.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const APP_URL = process.env.APP_URL ?? "http://127.0.0.1:3000";

// 125 administrative-task examples (scheduling, reminders, follow-ups, minutes,
// approvals, etc.) live in a sibling JSON file so the data is easy to edit and
// review separately from the seeding logic.
const EXAMPLES = JSON.parse(
  await readFile(fileURLToPath(new URL("./admin-examples.json", import.meta.url)), "utf8"),
);

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
