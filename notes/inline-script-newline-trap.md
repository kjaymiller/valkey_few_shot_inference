# The template-literal newline trap that made my button do nothing

> Draft note for a future blog post. Real bug, hit while building a tiny test
> UI served straight from a Fermyon Spin TypeScript component.

## The symptom

I added a single-page test UI to a Spin component by serving an HTML string from
`GET /`. Click **Run inference** and... nothing. No network request, no error
toast, no console message I noticed at first. The button was simply dead. Every
other part of the page rendered fine.

Dead-on-click with *no* error is the tell: it usually means the click handler was
never attached, which usually means the whole `<script>` failed to parse.

## The cause

The browser JS lived inside a TypeScript **template literal**:

```ts
const INDEX_HTML = `<!doctype html>
...
<script>
  // ... my frontend code, including an SSE parser ...
  while ((sep = buf.indexOf("\n\n")) !== -1) {   // <-- the landmine
    const block = buf.slice(0, sep);
    for (const line of block.split("\n")) { ... }
  }
</script>`;
```

That `"\n\n"` looks like a JavaScript string containing two newline *escapes*.
But it isn't browser JS yet — it's text inside a template literal. The template
literal is processed **first**, and a template literal turns `\n` into an actual
newline character. So by the time the string is sent to the browser, the served
HTML literally contains:

```js
  while ((sep = buf.indexOf("
")) !== -1) {
```

A raw line break in the middle of a string literal is a `SyntaxError`. One bad
line poisons the entire `<script>`, so none of the handlers attach — and the
browser reports it quietly in the console while the page looks otherwise normal.

## Why the compiler didn't catch it

To `tsc` and `esbuild`, the code is flawless: a valid string (`"\n\n"`) nested
inside a valid string (the template literal). The bug only exists in the
*rendered output*, a layer the type checker never looks at. Same reason it
survives bundling untouched.

## The fix

Escape the backslash so it survives the template-literal pass and reaches the
browser as a real `\n` escape:

```ts
while ((sep = buf.indexOf("\\n\\n")) !== -1) {
  for (const line of block.split("\\n")) { ... }
}
```

Rule of thumb: **any backslash escape meant for the inner (browser) JS needs to
be doubled inside the outer template literal.** `\n` → `\\n`, `\t` → `\\t`,
`\\` → `\\\\`.

## How to never ship it again

The compiler can't see it, so check the thing the browser actually receives.
A tiny build step: pull the `INDEX_HTML` template out of source, evaluate it as
a template literal (reproducing the runtime transform), extract the `<script>`,
and syntax-check that with Node's `vm`:

```js
import vm from "node:vm";
const m = src.match(/const INDEX_HTML\s*=\s*`([\s\S]*?)`;/);
const html = vm.runInNewContext("`" + m[1] + "`");   // resolve escapes like runtime
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);                               // parse-only; throws on the bug
```

Wire it into `build` before the bundle step and a broken inline script fails the
build with a clear message instead of silently shipping a dead page. (See
`app/scripts/check-frontend.mjs` in this repo.)

## Takeaways

- A button that does nothing with no error almost always means the script didn't
  parse and handlers never bound — check the console first.
- Embedding one language inside another's string literal stacks two escaping
  layers. The inner escapes have to survive the outer parse.
- If a class of bug is invisible to your type checker, add a check that inspects
  the *output*, not the source.
