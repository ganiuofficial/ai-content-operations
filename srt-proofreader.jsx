import React, { useState, useRef, useCallback } from "react";
import { Upload, CheckCircle2, AlertTriangle, Download, Loader2, FileText, RotateCcw, Sparkles, Copy, Check, ClipboardPaste } from "lucide-react";

const CHUNK_SIZE = 10;
const MODEL = "claude-sonnet-4-6";
const MAX_ATTEMPTS = 3;
const TONES = ["Professional", "Educational", "Promotional", "Inspirational", "Executive", "Conversational"];
const AUDIENCES = ["General public", "Startups", "Corporate", "Non-profits"];

// ---------- helpers ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseSRT(raw) {
  const blocks = raw.replace(/\r\n/g, "\n").trim().split(/\n\s*\n/);
  const subs = [];
  for (const block of blocks) {
    const lines = block.trim().split("\n");
    if (lines.length < 2) continue;
    const id = lines[0].trim();
    const timestamp = lines[1].trim();
    const text = lines.slice(2).join("\n");
    if (id && timestamp.includes("-->")) subs.push({ id, timestamp, text });
  }
  return subs;
}

function buildSRT(subs, corrections) {
  return subs
    .map((s) => `${s.id}\n${s.timestamp}\n${corrections[s.id] !== undefined ? corrections[s.id] : s.text}`)
    .join("\n\n") + "\n";
}

function chunkArray(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

function wordDiff(a, b) {
  const aw = a.split(/(\s+)/);
  const bw = b.split(/(\s+)/);
  const m = aw.length, n = bw.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = aw[i] === bw[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (aw[i] === bw[j]) { out.push({ type: "same", text: aw[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ type: "del", text: aw[i] }); i++; }
    else { out.push({ type: "add", text: bw[j] }); j++; }
  }
  while (i < m) out.push({ type: "del", text: aw[i++] });
  while (j < n) out.push({ type: "add", text: bw[j++] });
  return out;
}

function stripFences(text) {
  return text.trim().replace(/^```json\s*/i, "").replace(/^```\s*/, "").replace(/```\s*$/, "").trim();
}

function extractStyleHints(correctionsMap) {
  const allText = Object.values(correctionsMap).join(" ");
  const capWords = {};
  (allText.match(/\b[A-Z][a-z]{2,}\b/g) || []).forEach((w) => (capWords[w] = (capWords[w] || 0) + 1));
  const hyphens = {};
  (allText.match(/\b\w+-\w+\b/g) || []).forEach((w) => (hyphens[w] = (hyphens[w] || 0) + 1));
  const topCap = Object.entries(capWords).sort((a, b) => b[1] - a[1]).slice(0, 15).map(([w]) => w);
  const topHyphen = Object.entries(hyphens).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([w]) => w);
  return { topCap, topHyphen };
}

async function callClaudeJSON({ system, user, maxTokens = 1000 }) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, max_tokens: maxTokens, system, messages: [{ role: "user", content: user }] }),
  });
  if (!response.ok) throw new Error(`API error ${response.status}`);
  const data = await response.json();
  const textBlock = data.content?.find((b) => b.type === "text");
  if (!textBlock) throw new Error("No text in response");
  return JSON.parse(stripFences(textBlock.text));
}

function buildProofSystemPrompt(language, protectedTerms, styleHints) {
  let s = `You are a meticulous professional subtitle proofreader working in strict ${language}. You correct spelling, punctuation, capitalization, grammar, subject-verb agreement, duplicated words, and obvious speech-recognition mistakes, without ever changing the speaker's meaning, tone, or natural voice. You do not rewrite for style.

Consistency rules to apply throughout:
- Numbers: write them as numerals (e.g. "10", not "ten"), applied consistently, unless a number naturally starts a sentence.
- Punctuation: keep a consistent style for quotation marks, dashes, and ellipses.
- Hyphenation: pick one spelling for compound/hyphenated words and keep it consistent (e.g. don't mix "e-mail" and "email").
- Capitalization: keep recurring proper nouns, titles, and terms capitalized the same way every time.`;
  if (protectedTerms?.length) s += `\n\nThe following terms must be preserved exactly as given, verbatim, even if they look unusual — do not "correct" them: ${protectedTerms.join(", ")}.`;
  if (styleHints?.topCap?.length || styleHints?.topHyphen?.length) {
    s += `\n\nStyle choices already established earlier in this same transcript — stay consistent with these forms:`;
    if (styleHints.topCap.length) s += `\nCapitalized terms used so far: ${styleHints.topCap.join(", ")}.`;
    if (styleHints.topHyphen.length) s += `\nHyphenated forms used so far: ${styleHints.topHyphen.join(", ")}.`;
  }
  s += `\n\nRespond with ONLY valid JSON, no markdown fences, no commentary, no explanation.`;
  return s;
}

async function proofreadChunk(chunk, language, protectedTerms, styleHints) {
  const items = chunk.map((s) => ({ id: s.id, text: s.text }));
  const system = buildProofSystemPrompt(language, protectedTerms, styleHints);
  const user = `Correct the "text" field of each item below. Keep internal line breaks (\\n) where they aid readability, unless the split is clearly broken. If an item needs no changes, return it unchanged. Do not add or remove items, do not merge or split them.

Return JSON in exactly this shape:
{"items":[{"id":"<same id>","text":"<corrected text>"}],"flags":["<name or organization that should be verified, if any>"]}

INPUT:
${JSON.stringify(items)}`;
  const parsed = await callClaudeJSON({ system, user, maxTokens: 1000 });
  if (!parsed.items || !Array.isArray(parsed.items)) throw new Error("Malformed response shape");
  return parsed;
}

// ---------- content generation ----------
async function generateShortPack(sourceText, tone, audience) {
  const system = `You are a social media strategist turning source content into a ready-to-post bundle. Tone: ${tone}. Target audience: ${audience}. Adapt vocabulary, formality, and framing to that tone and audience while staying accurate to the source — no invented facts or statistics. Respond with ONLY valid JSON, no markdown fences, no commentary.`;
  const user = `Source content:\n"""${sourceText.slice(0, 6000)}"""\n\nProduce JSON in exactly this shape, keeping each field concise:
{
 "whatsapp_channel_post": "short punchy update, 2-4 sentences",
 "linkedin_post": "professional storytelling post, 100-150 words, closing call-to-action",
 "youtube_community_post": "casual short update, 2-3 sentences, can include a question",
 "instagram_post": "caption with line breaks and a few relevant emoji, 60-90 words, ending with 5-8 hashtags",
 "facebook_post": "conversational post, 60-90 words, ending with an open question",
 "x_post": "single post, STRICT MAXIMUM 200 characters including spaces",
 "youtube_title": "high-CTR title, under 70 characters",
 "youtube_description": "search-optimized description, 80-120 words, include a call to action, do not fabricate timestamps",
 "thumbnail_text_suggestions": ["3 to 4 short text overlay options, each under 5 words"]
}`;
  return callClaudeJSON({ system, user, maxTokens: 1000 });
}

async function generateLongPack(sourceText, tone, audience) {
  const system = `You are a content repurposing specialist turning source content into longer-form assets. Tone: ${tone}. Target audience: ${audience}. Stay accurate to the source — no invented facts or statistics. Keep each field within its word limit strictly, since output space is limited. Respond with ONLY valid JSON, no markdown fences, no commentary.`;
  const user = `Source content:\n"""${sourceText.slice(0, 6000)}"""\n\nProduce JSON in exactly this shape:
{
 "linkedin_article": "long-form LinkedIn article, 220-280 words, with a hook opening line, 2-3 short paragraphs, and a closing takeaway",
 "website_page_content": "webpage/blog body copy, 200-250 words, starting with a one-line H1-style title on its own first line, then body paragraphs",
 "tiktok_script": "TikTok script, 60-90 words: a spoken hook line, 2-3 short beats, and a closing line, written to be read aloud"
}`;
  return callClaudeJSON({ system, user, maxTokens: 1000 });
}

// ---------- reusable content pack panel ----------
function ContentPackResults({ pack, downloadPack }) {
  const [copiedKey, setCopiedKey] = useState("");
  const copy = (key, text) => {
    navigator.clipboard.writeText(text || "");
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(""), 1500);
  };
  const shortFields = [
    { key: "whatsapp_channel_post", label: "WhatsApp Channel Post" },
    { key: "linkedin_post", label: "LinkedIn Post (short)" },
    { key: "youtube_community_post", label: "YouTube Community Post" },
    { key: "instagram_post", label: "Instagram Post" },
    { key: "facebook_post", label: "Facebook Post" },
    { key: "x_post", label: "X Post (200 char max)" },
    { key: "youtube_title", label: "YouTube Title" },
    { key: "youtube_description", label: "YouTube Description" },
  ];
  const longFields = [
    { key: "linkedin_article", label: "LinkedIn Article (long-form)" },
    { key: "website_page_content", label: "Website Page Content" },
    { key: "tiktok_script", label: "TikTok Script" },
  ];
  const Field = ({ fkey, label }) => (
    <div className="border border-stone-200 rounded p-3">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">{label}</span>
        <button onClick={() => copy(fkey, pack[fkey])} className="flex items-center gap-1 text-xs text-teal-700 hover:text-teal-900">
          {copiedKey === fkey ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
        </button>
      </div>
      <p className="text-sm whitespace-pre-wrap text-stone-800">{pack[fkey]}</p>
      {fkey === "x_post" && (
        <p className={`text-xs mt-1 ${(pack.x_post || "").length > 200 ? "text-red-600" : "text-stone-400"}`}>{(pack.x_post || "").length}/200 characters</p>
      )}
    </div>
  );
  return (
    <div className="space-y-4">
      {shortFields.map((f) => <Field key={f.key} fkey={f.key} label={f.label} />)}
      {longFields.map((f) => <Field key={f.key} fkey={f.key} label={f.label} />)}
      <div className="border border-stone-200 rounded p-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">Thumbnail text suggestions</span>
        <ul className="text-sm mt-1.5 space-y-1 text-stone-800">
          {(pack.thumbnail_text_suggestions || []).map((t, i) => <li key={i}>• {t}</li>)}
        </ul>
      </div>
      <button onClick={downloadPack} className="flex items-center gap-2 px-4 py-2 border border-stone-300 rounded text-sm font-medium hover:bg-stone-100">
        <Download size={16} /> Download full pack (.txt)
      </button>
    </div>
  );
}

function RepurposePanel({ presetText, title, description, filenameBase }) {
  const [pastedText, setPastedText] = useState("");
  const [tone, setTone] = useState(TONES[0]);
  const [audience, setAudience] = useState(AUDIENCES[0]);
  const [status, setStatus] = useState("idle");
  const [pack, setPack] = useState(null);

  const sourceText = presetText !== undefined ? presetText : pastedText;

  const generate = useCallback(async () => {
    if (!sourceText || !sourceText.trim()) return;
    setStatus("processing");
    try {
      const [short, long] = await Promise.all([
        generateShortPack(sourceText, tone, audience),
        generateLongPack(sourceText, tone, audience),
      ]);
      setPack({ ...short, ...long });
      setStatus("done");
    } catch (err) {
      setStatus("error");
    }
  }, [sourceText, tone, audience]);

  const downloadPack = () => {
    if (!pack) return;
    const labels = {
      whatsapp_channel_post: "WHATSAPP CHANNEL POST", linkedin_post: "LINKEDIN POST",
      youtube_community_post: "YOUTUBE COMMUNITY POST", instagram_post: "INSTAGRAM POST",
      facebook_post: "FACEBOOK POST", x_post: "X POST (200 char max)",
      youtube_title: "YOUTUBE TITLE", youtube_description: "YOUTUBE DESCRIPTION",
      linkedin_article: "LINKEDIN ARTICLE", website_page_content: "WEBSITE PAGE CONTENT",
      tiktok_script: "TIKTOK SCRIPT",
    };
    let out = `CONTENT PACK\nTone: ${tone} | Audience: ${audience}\n\n`;
    Object.entries(labels).forEach(([key, label]) => { out += `--- ${label} ---\n${pack[key] || ""}\n\n`; });
    out += `--- THUMBNAIL TEXT SUGGESTIONS ---\n${(pack.thumbnail_text_suggestions || []).map((t) => `- ${t}`).join("\n")}\n`;
    const blob = new Blob([out], { type: "text/plain;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${filenameBase}_content_pack.txt`;
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  };

  return (
    <section className="bg-white border border-stone-200 rounded-md p-6">
      <h2 className="font-serif text-lg mb-1">{title}</h2>
      <p className="text-sm text-stone-500 mb-4">{description}</p>

      {presetText === undefined && (
        <textarea
          value={pastedText}
          onChange={(e) => setPastedText(e.target.value)}
          placeholder="Paste your existing content here — a transcript, article, notes, anything you want repurposed..."
          className="w-full h-36 border border-stone-300 rounded p-3 text-sm font-mono mb-4"
        />
      )}

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="block text-xs font-medium text-stone-600 mb-1">Tone</label>
          <select value={tone} onChange={(e) => setTone(e.target.value)} className="border border-stone-300 rounded px-3 py-2 text-sm bg-white">
            {TONES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-stone-600 mb-1">Audience</label>
          <select value={audience} onChange={(e) => setAudience(e.target.value)} className="border border-stone-300 rounded px-3 py-2 text-sm bg-white">
            {AUDIENCES.map((a) => <option key={a}>{a}</option>)}
          </select>
        </div>
      </div>

      {status !== "processing" && (
        <button
          onClick={generate}
          disabled={!sourceText || !sourceText.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-teal-700 text-white rounded text-sm font-medium hover:bg-teal-800 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Sparkles size={16} /> {pack ? "Regenerate" : "Repurpose this content"}
        </button>
      )}
      {status === "processing" && (
        <div className="flex items-center gap-2 text-sm text-stone-600"><Loader2 size={16} className="animate-spin text-teal-700" /> Writing your content pack...</div>
      )}
      {status === "error" && (
        <div className="flex items-center gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3 mt-3">
          <AlertTriangle size={16} /> Something went wrong generating the pack — try again.
        </div>
      )}

      {pack && status === "done" && <div className="mt-5"><ContentPackResults pack={pack} downloadPack={downloadPack} /></div>}
    </section>
  );
}

// ---------- main component ----------
export default function SRTProofreader() {
  const [subs, setSubs] = useState([]);
  const [fileName, setFileName] = useState("");
  const [language, setLanguage] = useState("American English");
  const [protectedTermsInput, setProtectedTermsInput] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [log, setLog] = useState([]);
  const [corrections, setCorrections] = useState({});
  const [flags, setFlags] = useState([]);
  const [failedIds, setFailedIds] = useState([]);
  const fileInputRef = useRef(null);

  const reset = () => {
    setSubs([]); setFileName(""); setStatus("idle"); setProgress({ done: 0, total: 0 });
    setLog([]); setCorrections({}); setFlags([]); setFailedIds([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      setSubs(parseSRT(evt.target.result));
      setFileName(file.name);
      setStatus("idle");
      setCorrections({}); setFlags([]); setFailedIds([]); setLog([]);
    };
    reader.readAsText(file);
  };

  const appendLog = (msg) => setLog((l) => [...l, msg]);

  const start = useCallback(async () => {
    if (subs.length === 0) return;
    setStatus("processing");
    setLog([]);
    const protectedTerms = protectedTermsInput.split(",").map((t) => t.trim()).filter(Boolean);
    const chunks = chunkArray(subs, CHUNK_SIZE);
    setProgress({ done: 0, total: chunks.length });
    const localCorrections = {};
    const localFlags = [];
    let localFailed = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const styleHints = extractStyleHints(localCorrections);
      let ok = false;
      let lastErr = null;
      for (let attempt = 0; attempt < MAX_ATTEMPTS && !ok; attempt++) {
        if (attempt > 0) await sleep(600 * attempt);
        try {
          const result = await proofreadChunk(chunk, language, protectedTerms, styleHints);
          const returnedIds = new Set(result.items.map((it) => String(it.id)));
          const expectedIds = chunk.map((s) => String(s.id));
          const missing = expectedIds.filter((id) => !returnedIds.has(id));
          if (missing.length > 0) throw new Error(`Missing ids: ${missing.join(", ")}`);
          result.items.forEach((it) => { localCorrections[String(it.id)] = it.text; });
          if (result.flags?.length) localFlags.push(...result.flags);
          ok = true;
          appendLog(`Chunk ${i + 1}/${chunks.length} — proofread (lines ${chunk[0].id}–${chunk[chunk.length - 1].id})`);
        } catch (err) {
          lastErr = err;
        }
      }
      if (!ok) {
        chunk.forEach((s) => { localCorrections[s.id] = s.text; });
        localFailed.push(...chunk.map((s) => s.id));
        appendLog(`Chunk ${i + 1}/${chunks.length} — failed after ${MAX_ATTEMPTS} attempts (${lastErr?.message || "unknown error"}), kept original text`);
      }
      setProgress({ done: i + 1, total: chunks.length });
      setCorrections({ ...localCorrections });
      setFlags([...new Set(localFlags)]);
      setFailedIds([...localFailed]);
      await sleep(250);
    }

    if (localFailed.length > 0) {
      appendLog(`Retrying ${localFailed.length} failed line(s) in a final pass...`);
      const retrySubs = subs.filter((s) => localFailed.includes(s.id));
      const retryChunks = chunkArray(retrySubs, 5);
      const stillFailed = [];
      for (const rc of retryChunks) {
        await sleep(800);
        try {
          const styleHints = extractStyleHints(localCorrections);
          const result = await proofreadChunk(rc, language, protectedTerms, styleHints);
          const returnedIds = new Set(result.items.map((it) => String(it.id)));
          const expectedIds = rc.map((s) => String(s.id));
          const missing = expectedIds.filter((id) => !returnedIds.has(id));
          if (missing.length > 0) throw new Error("still missing ids");
          result.items.forEach((it) => { localCorrections[String(it.id)] = it.text; });
          if (result.flags?.length) localFlags.push(...result.flags);
          appendLog(`Recovered lines ${rc[0].id}–${rc[rc.length - 1].id} on retry`);
        } catch {
          stillFailed.push(...rc.map((s) => s.id));
        }
      }
      localFailed = stillFailed;
      setCorrections({ ...localCorrections });
      setFlags([...new Set(localFlags)]);
      setFailedIds([...localFailed]);
    }
    setStatus("done");
  }, [subs, language, protectedTermsInput]);

  const download = (content, filename) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  };

  const downloadSRT = () => {
    const base = fileName.replace(/\.srt$/i, "") || "subtitles";
    download(buildSRT(subs, corrections), `${base}_proofread.srt`);
  };

  const downloadReport = () => {
    const changed = subs.filter((s) => corrections[s.id] !== undefined && corrections[s.id] !== s.text);
    let report = `PROOFREADING REPORT\nFile: ${fileName}\nDialect: ${language}\nTotal subtitle lines: ${subs.length}\nLines changed: ${changed.length}\n\n`;
    report += flags.length ? `NAMES / ORGANIZATIONS TO VERIFY:\n${flags.map((f) => `- ${f}`).join("\n")}\n\n` : `NAMES / ORGANIZATIONS TO VERIFY: none flagged\n\n`;
    if (failedIds.length) report += `LINES THAT COULD NOT BE PROCESSED (original text kept, review manually):\n${failedIds.join(", ")}\n\n`;
    report += `CHANGED LINES:\n`;
    changed.forEach((s) => { report += `#${s.id} ${s.timestamp}\n- ${s.text.replace(/\n/g, " ")}\n+ ${corrections[s.id].replace(/\n/g, " ")}\n\n`; });
    const base = fileName.replace(/\.srt$/i, "") || "subtitles";
    download(report, `${base}_report.txt`);
  };

  const changedSubs = subs.filter((s) => corrections[s.id] !== undefined && corrections[s.id] !== s.text);
  const cleanedTranscript = subs.length ? subs.map((s) => corrections[s.id] ?? s.text).join(" ").replace(/\s+/g, " ").trim() : "";

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <div className="max-w-3xl mx-auto px-6 py-12 space-y-10">
        <header className="border-b border-stone-300 pb-6">
          <p className="text-xs tracking-widest uppercase text-teal-700 font-semibold mb-2">Subtitle Proofreading Desk</p>
          <h1 className="font-serif text-3xl text-stone-900">SRT Proofreader & Content Repurposer</h1>
          <p className="text-stone-600 mt-2 text-sm leading-relaxed">
            Proofread a subtitle file with locked-in numbers and timestamps, or repurpose any content you already have — no video required.
          </p>
        </header>

        {/* ---------- SRT PROOFREADING ---------- */}
        <div>
          <h2 className="font-serif text-xl mb-4">1. Proofread a subtitle file</h2>
          <section className="bg-white border border-stone-200 rounded-md p-6 mb-6">
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-2 px-4 py-2 bg-stone-900 text-white rounded cursor-pointer hover:bg-stone-800 text-sm font-medium">
                <Upload size={16} /> Choose .srt file
                <input ref={fileInputRef} type="file" accept=".srt" onChange={handleFile} className="hidden" />
              </label>
              <select value={language} onChange={(e) => setLanguage(e.target.value)} className="border border-stone-300 rounded px-3 py-2 text-sm bg-white" disabled={status === "processing"}>
                <option>American English</option>
                <option>British English</option>
              </select>
              {subs.length > 0 && status !== "processing" && (
                <button onClick={reset} className="flex items-center gap-1 text-sm text-stone-500 hover:text-stone-800"><RotateCcw size={14} /> Start over</button>
              )}
            </div>

            {fileName && (
              <div className="mt-4 flex items-center gap-2 text-sm text-stone-700">
                <FileText size={16} className="text-teal-700" />
                <span className="font-medium">{fileName}</span>
                <span className="text-stone-400">— {subs.length} subtitle lines</span>
              </div>
            )}

            {subs.length > 0 && status === "idle" && (
              <div className="mt-5">
                <label className="block text-xs font-medium text-stone-600 mb-1.5">Terms to keep exactly as-is (names, brands, jargon) — comma separated, optional</label>
                <input type="text" value={protectedTermsInput} onChange={(e) => setProtectedTermsInput(e.target.value)} placeholder="e.g. Adaeze, Zenlytics, Ibadan" className="w-full border border-stone-300 rounded px-3 py-2 text-sm mb-4" />
                <button onClick={start} className="px-5 py-2.5 bg-teal-700 text-white rounded text-sm font-medium hover:bg-teal-800">Proofread all {subs.length} lines</button>
              </div>
            )}
          </section>

          {status !== "idle" && (
            <section className="bg-white border border-stone-200 rounded-md p-6 mb-6">
              <div className="flex items-center gap-2 mb-3">
                {status === "processing" && <Loader2 size={16} className="animate-spin text-teal-700" />}
                {status === "done" && <CheckCircle2 size={16} className="text-teal-700" />}
                <span className="text-sm font-medium">{status === "processing" ? `Processing chunk ${progress.done}/${progress.total}...` : `Done — ${progress.total} chunks processed`}</span>
              </div>
              <div className="w-full bg-stone-100 rounded-full h-1.5 mb-4">
                <div className="bg-teal-700 h-1.5 rounded-full transition-all" style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
              </div>
              <div className="font-mono text-xs text-stone-500 max-h-40 overflow-y-auto space-y-1">{log.map((l, i) => <div key={i}>{l}</div>)}</div>
            </section>
          )}

          {status === "done" && (
            <>
              <section className="bg-white border border-stone-200 rounded-md p-6 mb-6">
                <h3 className="font-serif text-lg mb-4">Results</h3>
                <div className="grid grid-cols-3 gap-4 mb-5 text-center">
                  <div className="border border-stone-200 rounded p-3"><div className="text-2xl font-semibold text-stone-900">{subs.length}</div><div className="text-xs text-stone-500 mt-1">Total lines</div></div>
                  <div className="border border-stone-200 rounded p-3"><div className="text-2xl font-semibold text-teal-700">{changedSubs.length}</div><div className="text-xs text-stone-500 mt-1">Lines changed</div></div>
                  <div className="border border-stone-200 rounded p-3"><div className="text-2xl font-semibold text-amber-600">{flags.length}</div><div className="text-xs text-stone-500 mt-1">Flags to verify</div></div>
                </div>
                {failedIds.length > 0 && (
                  <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded p-3 mb-5 text-sm text-amber-800">
                    <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                    <span>{failedIds.length} line(s) still could not be processed after retries and were kept as original text: {failedIds.join(", ")}. Review these manually.</span>
                  </div>
                )}
                <div className="flex gap-3 flex-wrap">
                  <button onClick={downloadSRT} className="flex items-center gap-2 px-4 py-2 bg-stone-900 text-white rounded text-sm font-medium hover:bg-stone-800"><Download size={16} /> Download proofread .srt</button>
                  <button onClick={downloadReport} className="flex items-center gap-2 px-4 py-2 border border-stone-300 rounded text-sm font-medium hover:bg-stone-100"><Download size={16} /> Download report (.txt)</button>
                </div>
              </section>

              {flags.length > 0 && (
                <section className="bg-white border border-stone-200 rounded-md p-6 mb-6">
                  <h3 className="font-serif text-lg mb-3">Flagged for verification</h3>
                  <ul className="text-sm space-y-1 text-stone-700">{flags.map((f, i) => <li key={i} className="flex items-center gap-2"><AlertTriangle size={14} className="text-amber-600" /> {f}</li>)}</ul>
                </section>
              )}

              {changedSubs.length > 0 && (
                <section className="bg-white border border-stone-200 rounded-md p-6 mb-6">
                  <h3 className="font-serif text-lg mb-4">Changed lines ({changedSubs.length})</h3>
                  <div className="space-y-4 max-h-[32rem] overflow-y-auto">
                    {changedSubs.map((s) => {
                      const diff = wordDiff(s.text.replace(/\n/g, " "), corrections[s.id].replace(/\n/g, " "));
                      return (
                        <div key={s.id} className="border-b border-stone-100 pb-3">
                          <div className="text-xs text-stone-400 font-mono mb-1">#{s.id} · {s.timestamp}</div>
                          <div className="font-mono text-sm leading-relaxed">
                            {diff.map((d, i) => d.type === "same" ? <span key={i}>{d.text}</span> : d.type === "del" ? <span key={i} className="line-through text-red-500 bg-red-50">{d.text}</span> : <span key={i} className="text-teal-700 bg-teal-50 underline">{d.text}</span>)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>
              )}

              <RepurposePanel
                presetText={cleanedTranscript}
                title="Repurpose this transcript"
                description="Built from your proofread subtitles — set a tone and audience, then generate everything below."
                filenameBase={(fileName || "subtitles").replace(/\.srt$/i, "")}
              />
            </>
          )}
        </div>

        {/* ---------- STANDALONE REPURPOSING ---------- */}
        <div>
          <h2 className="font-serif text-xl mb-1 flex items-center gap-2"><ClipboardPaste size={20} className="text-teal-700" /> 2. Repurpose existing content — no subtitles needed</h2>
          <p className="text-sm text-stone-500 mb-4">Have old content with no video or .srt file? Paste it in and repurpose it directly.</p>
          <RepurposePanel title="Paste your content" description="Any transcript, article, or notes — set tone and audience, then repurpose." filenameBase="repurposed_content" />
        </div>
      </div>
    </div>
  );
}
