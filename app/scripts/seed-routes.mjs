#!/usr/bin/env node
// Seed the semantic router by POSTing labeled exemplar utterances to the running
// Spin app's /routes endpoint. The app handles embedding + indexing into
// idx:routes, so this script needs no Valkey client of its own.
//
// Usage:
//   spin up --variable valkey_url="$VALKEY_URL"   # in another terminal
//   APP_URL=http://127.0.0.1:3000 node scripts/seed-routes.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const APP_URL = process.env.APP_URL ?? "http://127.0.0.1:3000";

// Labeled exemplars (a few utterances per intent) live in a sibling JSON file
// so the route definitions are easy to edit and review separately.
const ROUTES = JSON.parse(
  await readFile(fileURLToPath(new URL("./routes.json", import.meta.url)), "utf8"),
);

async function main() {
  console.log(`Seeding ${ROUTES.length} route exemplars to ${APP_URL}/routes ...`);
  for (const r of ROUTES) {
    const res = await fetch(`${APP_URL}/routes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(r),
    });
    const text = await res.text();
    if (!res.ok) {
      console.error(`  ✗ [${r.route}] ${r.utterance.slice(0, 32)}… -> ${res.status} ${text}`);
      process.exitCode = 1;
    } else {
      console.log(`  ✓ [${r.route}] ${r.utterance.slice(0, 32)}…`);
    }
  }
  const labels = [...new Set(ROUTES.map((r) => r.route))];
  console.log(`Done. ${labels.length} routes: ${labels.join(", ")}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
