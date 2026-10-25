/**
 * A tiny Python highlighter for the "How this was calculated" panel: one regex pass, no dependency.
 * Returns the source split into lines of tokens; a token never spans a line break.
 */
export type Tok = { k: 'kw' | 'str' | 'com' | 'num' | 'deco' | 'fn' | 'const' | 'txt'; v: string };

const KW = new Set(('and as assert async await break class continue def del elif else except finally for from global if import ' +
  'in is lambda nonlocal not or pass raise return try while with yield').split(' '));
const CONST = new Set(['True', 'False', 'None', 'self', 'cls']);
const RE = new RegExp([
  /(#[^\n]*)/.source,
  /([rRbBuUfF]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))/.source,
  /(\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b)/.source,
  /(@[\w.]+)/.source,
  /([A-Za-z_]\w*)/.source,
  /([\s\S])/.source,
].join('|'), 'g');

export function highlightPython(src: string): Tok[][] {
  const toks: Tok[] = [];
  let prev = '';
  for (const m of src.matchAll(RE)) {
    const [v, com, str, num, deco, id] = m;
    let k: Tok['k'] = 'txt';
    if (com) k = 'com';
    else if (str) k = 'str';
    else if (num) k = 'num';
    else if (deco) k = 'deco';
    else if (id) k = KW.has(id) ? 'kw' : CONST.has(id) ? 'const' : prev === 'def' || prev === 'class' ? 'fn' : 'txt';
    if (id || com || str || num || deco) prev = v;
    const last = toks[toks.length - 1];
    if (k === 'txt' && last?.k === 'txt') last.v += v; // merge plain runs
    else toks.push({ k, v });
  }
  const lines: Tok[][] = [[]];
  for (const t of toks) {
    t.v.split('\n').forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1]!.push({ k: t.k, v: part });
    });
  }
  if (src.endsWith('\n')) lines.pop();
  return lines;
}
