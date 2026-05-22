import Link from "next/link";
import DemoLaunchButton from "@/components/DemoLaunchButton";
import {
  FiTrendingUp,
  FiArrowRight,
  FiGithub,
  FiLayers,
  FiFilter,
  FiShield,
  FiZap,
  FiFileText,
  FiActivity,
  FiCheckCircle,
  FiSearch,
  FiCpu,
} from "react-icons/fi";

const GITHUB_URL = "https://github.com/PatelDeep223/finbot-financial-rag";

const STATS = [
  { value: "~15ms", label: "Cached response" },
  { value: "10-stage", label: "RAG pipeline" },
  { value: "90", label: "Tests passing" },
  { value: "500-page", label: "Documents supported" },
];

const PROBLEMS = [
  {
    problem: "Exact numbers get missed",
    detail: "Vector search finds “revenue grew” but skips the exact “$4.2B” figure that actually matters.",
    solution: "Hybrid BM25 + FAISS retrieval — keyword search catches exact figures, vector search catches meaning.",
  },
  {
    problem: "Irrelevant chunks rank high",
    detail: "Plain vector search returns 5 chunks where 2 are noise — disclaimers, headers, page footers.",
    solution: "Cross-encoder reranker rescores 20 candidates and keeps only the 5 most relevant.",
  },
  {
    problem: "The model invents figures",
    detail: "An LLM confidently answers “$4.3B” when the real number in the filing is “$4.2B”.",
    solution: "4-layer anti-hallucination — temperature 0, strict prompts, phrase detection, source verification.",
  },
  {
    problem: "Repeat queries are slow",
    detail: "The same question runs the full pipeline every time — 3 to 5 seconds, every time.",
    solution: "Redis semantic cache returns repeated answers in ~15ms instead of 3–5s.",
  },
];

const FEATURES = [
  {
    icon: FiLayers,
    title: "Hybrid retrieval",
    desc: "BM25 keyword search and FAISS vector search run in parallel, fused with Reciprocal Rank Fusion.",
  },
  {
    icon: FiFilter,
    title: "Cross-encoder reranking",
    desc: "ms-marco-MiniLM rescores 20 candidates down to the 5 most relevant — fast and precise.",
  },
  {
    icon: FiShield,
    title: "4-layer anti-hallucination",
    desc: "Strict prompts, deterministic output, uncertainty detection and source checks on every answer.",
  },
  {
    icon: FiZap,
    title: "Redis semantic cache",
    desc: "Repeated questions return in ~15ms with a 40–60% hit rate, with a graceful local fallback.",
  },
  {
    icon: FiFileText,
    title: "Source citations",
    desc: "Every answer cites the document name, page number and a relevance score — fully accountable.",
  },
  {
    icon: FiActivity,
    title: "Real-time streaming",
    desc: "Answers stream token-by-token over Server-Sent Events for an instant, responsive feel.",
  },
];

const STEPS = [
  { icon: FiZap, title: "Cache check", desc: "Redis semantic cache — instant 15ms answer on a hit." },
  { icon: FiSearch, title: "Route & rewrite", desc: "Classify intent, rewrite vague queries for retrieval." },
  { icon: FiFilter, title: "Search & rerank", desc: "Hybrid search, then cross-encoder picks the top 5." },
  { icon: FiCpu, title: "Grounded answer", desc: "GPT-3.5 at temperature 0 answers only from context." },
  { icon: FiCheckCircle, title: "Verify & cite", desc: "Confidence score plus document and page citations." },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white text-slate-900">
      {/* ── Navbar ─────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 bg-white/85 backdrop-blur-md border-b border-slate-200">
        <nav className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
              <FiTrendingUp className="w-5 h-5" />
            </div>
            <span className="text-lg font-semibold tracking-tight">FinBot</span>
          </Link>

          <div className="hidden md:flex items-center gap-8 text-sm text-slate-600">
            <a href="#problem" className="hover:text-slate-900 transition">Why FinBot</a>
            <a href="#features" className="hover:text-slate-900 transition">Features</a>
            <a href="#pipeline" className="hover:text-slate-900 transition">How it works</a>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/login" className="hidden sm:block text-sm text-slate-600 hover:text-slate-900 transition">
              Sign in
            </Link>
            <DemoLaunchButton className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition">
              Launch app <FiArrowRight className="w-4 h-4" />
            </DemoLaunchButton>
          </div>
        </nav>
      </header>

      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-16 grid lg:grid-cols-2 gap-14 items-center">
        <div className="animate-fade-up">
          <span className="inline-flex items-center gap-2 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-full px-3 py-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            Production-grade Financial RAG
          </span>

          <h1 className="mt-6 text-4xl sm:text-5xl font-serif leading-[1.1] tracking-tight">
            Ask your financial documents anything —
            <span className="text-emerald-600"> and trust every answer.</span>
          </h1>

          <p className="mt-5 text-lg text-slate-600 leading-relaxed max-w-xl">
            FinBot is a retrieval system built for earnings reports, 10-Ks and filings.
            Hybrid retrieval, cross-encoder reranking and 4-layer anti-hallucination —
            so the numbers come back right, with a source for every claim.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <DemoLaunchButton className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-6 py-3 rounded-lg transition">
              Try the live demo <FiArrowRight className="w-4 h-4" />
            </DemoLaunchButton>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 border border-slate-300 hover:border-slate-400 text-slate-700 font-medium px-6 py-3 rounded-lg transition"
            >
              <FiGithub className="w-4 h-4" /> View source
            </a>
          </div>

          <p className="mt-6 text-sm text-slate-400">
            10-stage pipeline · 90 tests passing · ~15ms cached responses
          </p>
        </div>

        {/* Hero mockup — chat preview */}
        <div className="animate-fade-up" style={{ animationDelay: "0.12s" }}>
          <div className="rounded-2xl border border-slate-200 bg-white shadow-xl shadow-slate-200/60 overflow-hidden">
            <div className="flex items-center gap-1.5 px-4 py-3 border-b border-slate-100 bg-slate-50">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
              <span className="ml-2 text-xs text-slate-400 font-mono">finbot · chat</span>
            </div>

            <div className="p-5 space-y-4">
              {/* user */}
              <div className="flex justify-end">
                <div className="bg-emerald-50 border border-emerald-200 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-slate-700 max-w-[80%]">
                  What was Q3 2024 revenue, and how did it grow?
                </div>
              </div>

              {/* bot */}
              <div className="flex gap-2.5">
                <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center text-white shrink-0 mt-1">
                  <FiCpu className="w-3.5 h-3.5" />
                </div>
                <div className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 text-sm max-w-[85%]">
                  <p className="text-emerald-700 text-[11px] font-semibold uppercase tracking-wider mb-1">
                    Answer
                  </p>
                  <p className="text-slate-700 leading-relaxed">
                    Q3 2024 revenue was{" "}
                    <span className="font-semibold text-slate-900">$4.2B</span>, up{" "}
                    <span className="font-semibold text-slate-900">18% YoY</span>, driven
                    by cloud and subscription growth.
                  </p>
                  <div className="mt-3 flex items-center gap-2 text-[11px]">
                    <span className="inline-flex items-center gap-1 bg-white border border-slate-200 rounded-md px-2 py-1 text-slate-500">
                      <FiFileText className="w-3 h-3" /> NovaTech Q3 2024 · p.3
                    </span>
                    <span className="inline-flex items-center gap-1 bg-emerald-50 border border-emerald-200 rounded-md px-2 py-1 text-emerald-700">
                      <FiCheckCircle className="w-3 h-3" /> 98% confident
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats bar ──────────────────────────────────────────── */}
      <section className="border-y border-slate-200 bg-slate-50">
        <div className="max-w-6xl mx-auto px-6 py-10 grid grid-cols-2 lg:grid-cols-4 gap-6">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-serif text-emerald-600">{s.value}</div>
              <div className="mt-1 text-sm text-slate-500">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Problem → Solution ─────────────────────────────────── */}
      <section id="problem" className="max-w-6xl mx-auto px-6 py-20">
        <div className="max-w-2xl">
          <h2 className="text-3xl sm:text-4xl font-serif tracking-tight">
            Naive RAG fails on financial documents
          </h2>
          <p className="mt-4 text-slate-600 leading-relaxed">
            Standard “embed → retrieve → generate” pipelines lose the exact figures finance
            depends on. FinBot is engineered to fix each failure mode.
          </p>
        </div>

        <div className="mt-12 grid md:grid-cols-2 gap-5">
          {PROBLEMS.map((p) => (
            <div
              key={p.problem}
              className="rounded-xl border border-slate-200 bg-white p-6 hover:border-slate-300 transition"
            >
              <h3 className="font-semibold text-slate-900">{p.problem}</h3>
              <p className="mt-2 text-sm text-slate-500 leading-relaxed">{p.detail}</p>
              <div className="mt-4 flex gap-2.5 rounded-lg bg-emerald-50 border border-emerald-100 p-3">
                <FiCheckCircle className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <p className="text-sm text-emerald-800 leading-relaxed">{p.solution}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────── */}
      <section id="features" className="bg-slate-50 border-y border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="max-w-2xl">
            <h2 className="text-3xl sm:text-4xl font-serif tracking-tight">
              Built like a production system
            </h2>
            <p className="mt-4 text-slate-600 leading-relaxed">
              Every stage of the pipeline earns its place — retrieval, ranking, generation
              and verification, all measured and tested.
            </p>
          </div>

          <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="rounded-xl border border-slate-200 bg-white p-6 hover:shadow-md hover:shadow-slate-200/70 transition"
              >
                <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600">
                  <f.icon className="w-5 h-5" />
                </div>
                <h3 className="mt-4 font-semibold text-slate-900">{f.title}</h3>
                <p className="mt-2 text-sm text-slate-500 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ───────────────────────────────────────── */}
      <section id="pipeline" className="max-w-6xl mx-auto px-6 py-20">
        <div className="max-w-2xl">
          <h2 className="text-3xl sm:text-4xl font-serif tracking-tight">
            How a question becomes a trusted answer
          </h2>
          <p className="mt-4 text-slate-600 leading-relaxed">
            Five stages, distilled from the full 10-stage pipeline — from cache lookup to a
            cited, confidence-scored answer.
          </p>
        </div>

        <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {STEPS.map((s, i) => (
            <div key={s.title} className="relative rounded-xl border border-slate-200 bg-white p-5">
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-lg bg-slate-900 flex items-center justify-center text-white">
                  <s.icon className="w-4 h-4" />
                </div>
                <span className="text-2xl font-serif text-slate-200">{i + 1}</span>
              </div>
              <h3 className="mt-3 font-semibold text-sm text-slate-900">{s.title}</h3>
              <p className="mt-1.5 text-xs text-slate-500 leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Final CTA ──────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <div className="rounded-2xl bg-slate-900 px-8 py-16 text-center">
          <h2 className="text-3xl sm:text-4xl font-serif text-white tracking-tight">
            See FinBot answer your documents
          </h2>
          <p className="mt-4 text-slate-300 max-w-xl mx-auto leading-relaxed">
            Upload a financial PDF, ask a question, and watch the answer stream in with
            sources and a confidence score.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <DemoLaunchButton className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-6 py-3 rounded-lg transition">
              Launch the app <FiArrowRight className="w-4 h-4" />
            </DemoLaunchButton>
            <Link
              href="/signup"
              className="inline-flex items-center gap-2 border border-slate-600 hover:border-slate-400 text-white font-medium px-6 py-3 rounded-lg transition"
            >
              Create an account
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-10 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
              <FiTrendingUp className="w-4 h-4" />
            </div>
            <span className="text-sm text-slate-500">
              FinBot — Financial Intelligence RAG
            </span>
          </div>
          <div className="flex items-center gap-6 text-sm text-slate-500">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 hover:text-slate-900 transition">
              <FiGithub className="w-4 h-4" /> GitHub
            </a>
            <span>Built by Deep Patel</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
