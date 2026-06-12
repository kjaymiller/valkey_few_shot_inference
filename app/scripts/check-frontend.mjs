#!/usr/bin/env node
// Build-time guard for the inline browser JS served from GET / (the INDEX_HTML
// template literal in src/index.ts).
//
// Why this exists: the page's <script> is written inside a TS template literal,
// so a string like "\n\n" in the browser JS is consumed by the template literal
// as a REAL newline and lands as a raw line break inside a JS string in the
// served HTML — a SyntaxError that kills the whole <script>, so no handlers
// attach and the UI silently does nothing. tsc/esbuild can't see it: to them
// it's just a valid string inside a valid string. The only way to catch it is
// to reproduce the browser's view and parse THAT.
//
// We extract the INDEX_HTML template, evaluate it the way the runtime will
// (resolving escapes exactly like a template literal), pull out the <script>,
// and syntax-check it with vm.Script. Fails the build on a parse error.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const src = await readFile(
  fileURLToPath(new URL("../src/index.ts", import.meta.url)),
  "utf8",
);

// Grab the INDEX_HTML template literal body (between the first pair of backticks
// after its declaration). The HTML itself contains no backticks, so this is safe.
const m = src.match(/const INDEX_HTML\s*=\s*`([\s\S]*?)`;/);
if (!m) {
  console.error("check-frontend: could not find INDEX_HTML template in src/index.ts");
  process.exit(1);
}

// Re-evaluate the captured text AS a template literal so backslash escapes
// (\n, \t, \\) resolve to the exact bytes the browser will receive — the same
// transformation the running component applies. No interpolation is used in
// INDEX_HTML, so an empty scope is fine.
let html;
try {
  html = vm.runInNewContext("`" + m[1] + "`");
} catch (e) {
  console.error("check-frontend: INDEX_HTML is not a valid template literal:", e.message);
  process.exit(1);
}

const sm = html.match(/<script>([\s\S]*?)<\/script>/);
if (!sm) {
  console.error("check-frontend: no <script> block found in rendered HTML");
  process.exit(1);
}

try {
  // Parse-only: never executes the browser code, just validates its syntax.
  new vm.Script(sm[1], { filename: "INDEX_HTML <script>" });
} catch (e) {
  console.error("check-frontend: inline <script> has a syntax error:\n  " + e.message);
  console.error("  (likely an unescaped \\n / \\t — double it to \\\\n inside the template literal)");
  process.exit(1);
}

console.log("check-frontend: inline <script> parses cleanly ✓");
