"use client";

import { useMemo, useState } from "react";

type NavPage = "Analyze" | "Model Insights" | "How It Works" | "About Project";
type Feature = { feature: string; tfidf: number; coefficient: number; contribution: number };
type ScoreItem = { label: string; display_label: string; score: number };
type PredictResult = {
  top_label: string;
  top_display_label: string;
  top_score: number;
  second_label: string;
  second_display_label: string;
  second_score: number;
  margin: number;
  clarity: "uncertain" | "relatively_clear" | "clearer";
  uncertain: boolean;
  scores: ScoreItem[];
  explanation: { supporting: Feature[]; opposing: Feature[] };
};

const examples: Record<string, string> = {
  "Choose an example": "",
  "Neutral daily-life text":
    "Today was a normal day. I went to class, finished my work, and talked with my friends.",
  "Mild negative mood":
    "I have been feeling a little sad lately, but I can still do my daily activities.",
  "Daily-life impact":
    "For the past few weeks, I have lost interest in things I used to enjoy and it is becoming difficult to focus on my work.",
  "High-risk language example":
    "I feel completely hopeless and I keep thinking that life is not worth living.",
};

const classMeta: Record<string, { subtitle: string; color: string }> = {
  Minimal: { subtitle: "lowest dataset label", color: "#10a37f" },
  Mild: { subtitle: "mild dataset label", color: "#5b7cfa" },
  Moderate: { subtitle: "moderate dataset label", color: "#d98900" },
  Severe: { subtitle: "highest dataset label", color: "#d85a66" },
};

const distribution = [
  { label: "Minimal", count: 2568, share: 72.67 },
  { label: "Mild", count: 290, share: 8.21 },
  { label: "Moderate", count: 394, share: 11.15 },
  { label: "Severe", count: 282, share: 7.98 },
];

const modelComparison = [
  { name: "TF-IDF Bigram + Balanced LR", macroF1: 45.60 },
  { name: "BoW + Balanced LR", macroF1: 42.27 },
  { name: "BoW Bigram + Balanced LR", macroF1: 41.34 },
  { name: "BoW + Logistic Regression", macroF1: 39.73 },
  { name: "Dummy majority", macroF1: 21.05 },
];

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function clarityTitle(value: PredictResult["clarity"]) {
  if (value === "uncertain") return "Uncertain";
  if (value === "clearer") return "Clearer prediction";
  return "Relatively clear";
}

export default function Home() {
  const [page, setPage] = useState<NavPage>("Analyze");
  const [selectedExample, setSelectedExample] = useState("Choose an example");
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resultTab, setResultTab] = useState<"score" | "why">("score");

  const scoreMap = useMemo(() => {
    const map = new Map<string, number>();
    result?.scores.forEach((item) => map.set(item.display_label, item.score));
    return map;
  }, [result]);

  function chooseExample(name: string) {
    setSelectedExample(name);
    setText(examples[name] ?? "");
    setResult(null);
    setError("");
  }

  function clearAll() {
    setText("");
    setSelectedExample("Choose an example");
    setResult(null);
    setError("");
    setResultTab("score");
  }

  async function analyze() {
    if (!text.trim()) {
      setError("Please enter English text before analyzing.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Prediction request failed");
      setResult(payload);
      setResultTab("score");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze the text.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-mark">SC</div>
          <div className="brand-title">Severity Classifier</div>
          <div className="brand-subtitle">NLP • Machine Learning • Explainable AI</div>
        </div>

        <nav className="side-nav" aria-label="Main navigation">
          {(["Analyze", "Model Insights", "How It Works", "About Project"] as NavPage[]).map((item) => (
            <button
              className={page === item ? "nav-item active" : "nav-item"}
              key={item}
              onClick={() => setPage(item)}
            >
              <span className="nav-dot" />
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-meta">
          <Meta label="Model" value="TF-IDF + Balanced Logistic Regression" />
          <Meta label="Language" value="English" />
          <Meta label="Version" value="Portfolio v3.0" />
          <Meta label="Runtime" value="Next.js + FastAPI" />
        </div>
      </aside>

      <section className="content">
        {page === "Analyze" && (
          <>
            <Hero
              kicker="Interactive NLP application"
              title={<>Depression Severity<br />Text Classifier</>}
              description="Explore how an explainable machine-learning model classifies English text into four dataset labels, reports how clearly the classes are separated, and shows which TF-IDF features influenced the result."
            />

            <div className="chips">
              <Chip>TF-IDF</Chip>
              <Chip>Unigram + Bigram</Chip>
              <Chip>Balanced Logistic Regression</Chip>
              <Chip>Explainable ML</Chip>
              <Chip>FastAPI</Chip>
            </div>

            <div className="notice">
              <b>Educational use only.</b> This application analyzes text patterns with machine learning. It is not a medical diagnosis or clinical-risk assessment tool.
            </div>

            <section className="section-block">
              <div className="section-kicker">Analyze</div>
              <h2>Analyze an English text</h2>
              <p className="section-copy">Enter or select an example to view class scores, prediction clarity and model explanation.</p>

              <label className="field-label" htmlFor="example">Example</label>
              <select id="example" value={selectedExample} onChange={(e) => chooseExample(e.target.value)}>
                {Object.keys(examples).map((name) => <option key={name}>{name}</option>)}
              </select>

              <label className="field-label" htmlFor="text">English text</label>
              <textarea
                id="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type or paste English text here..."
                maxLength={5000}
              />
              <div className="helper-row">
                <span>{text.length} characters • English text only</span>
                <span>Text is sent to the API only when you press Analyze</span>
              </div>

              <div className="action-row">
                <button className="button secondary" onClick={clearAll}>Clear</button>
                <button className="button primary" onClick={analyze} disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze text"}
                </button>
              </div>
              {error && <div className="error-box">{error}</div>}
            </section>

            {result && (
              <section className="result-section">
                <div className="section-kicker">Result</div>
                <h2>Analysis result</h2>

                <div className="result-grid">
                  <div className={result.uncertain ? "result-card uncertain" : "result-card"}>
                    <div className="status-pill">{clarityTitle(result.clarity)}</div>
                    <div className="result-label">{result.top_display_label}</div>
                    <p>
                      Highest model score is <b>{pct(result.top_score)}</b>. The top-two margin is <b>{pct(result.margin)}</b>.
                      {result.uncertain
                        ? " The classes are not separated enough for the demo to present the result as clear."
                        : " The leading class is separated from the alternatives more clearly under the demo rule."}
                    </p>
                  </div>
                  <KpiCard label="Highest model score" value={pct(result.top_score)} note="Highest score among the four model classes" />
                  <KpiCard label="Top-two margin" value={pct(result.margin)} note="Difference between the first and second scores" />
                </div>

                <h3>Class scores</h3>
                <div className="score-grid">
                  {["Minimal", "Mild", "Moderate", "Severe"].map((label) => {
                    const score = scoreMap.get(label) ?? 0;
                    const meta = classMeta[label];
                    return (
                      <div className="score-card" key={label}>
                        <div className="score-name">{label}</div>
                        <div className="score-value">{pct(score)}</div>
                        <div className="score-subtitle">{meta.subtitle}</div>
                        <div className="progress-track">
                          <div className="progress-fill" style={{ width: `${score * 100}%`, background: meta.color }} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="tabs">
                  <button className={resultTab === "score" ? "tab active" : "tab"} onClick={() => setResultTab("score")}>Score interpretation</button>
                  <button className={resultTab === "why" ? "tab active" : "tab"} onClick={() => setResultTab("why")}>Why this result?</button>
                </div>

                {resultTab === "score" ? (
                  <div className="tab-panel">
                    <h3>How to read this result</h3>
                    <p>
                      The model assigned the highest score to <b>{result.top_display_label}</b> at <b>{pct(result.top_score)}</b>.
                      The second-highest class is <b>{result.second_display_label}</b> at <b>{pct(result.second_score)}</b>, producing a margin of <b>{pct(result.margin)}</b>.
                    </p>
                    <div className="info-box">
                      Model scores compare labels inside this trained classifier. They are not probabilities of illness and the clarity rules are demo rules, not clinical thresholds.
                    </div>
                  </div>
                ) : (
                  <div className="tab-panel">
                    <h3>Features supporting {result.top_display_label}</h3>
                    <p className="section-copy">These features had positive local contribution to the linear score of the highest-scoring class.</p>
                    <FeatureList items={result.explanation.supporting} positive />

                    <details className="details-box">
                      <summary>Features reducing the {result.top_display_label} score</summary>
                      <FeatureList items={result.explanation.opposing} positive={false} />
                    </details>

                    <div className="warning-box">
                      Feature contribution = TF-IDF value × Logistic Regression coefficient. It explains the model calculation only and is not medical evidence.
                    </div>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {page === "Model Insights" && <ModelInsights />}
        {page === "How It Works" && <HowItWorks />}
        {page === "About Project" && <AboutProject />}

        <footer>Educational NLP portfolio project • Next.js + FastAPI • Model output is not a medical diagnosis</footer>
      </section>
    </main>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return <div className="meta-item"><span>{label}</span><b>{value}</b></div>;
}

function Hero({ kicker, title, description }: { kicker: string; title: React.ReactNode; description: string }) {
  return <header className="hero"><div className="hero-kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></header>;
}

function Chip({ children }: { children: React.ReactNode }) {
  return <span className="chip">{children}</span>;
}

function KpiCard({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="kpi-card"><span>{label}</span><strong>{value}</strong><p>{note}</p></div>;
}

function FeatureList({ items, positive }: { items: Feature[]; positive: boolean }) {
  if (!items.length) return <div className="empty-state">No active features were available for this direction.</div>;
  return (
    <div className="feature-list">
      {items.map((item) => (
        <div className="feature-row" key={`${item.feature}-${item.contribution}`}>
          <b>{item.feature}</b>
          <span className={positive ? "feature-positive" : "feature-negative"}>{item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

function ModelInsights() {
  const maxDistribution = Math.max(...distribution.map((item) => item.count));
  const maxF1 = Math.max(...modelComparison.map((item) => item.macroF1));
  return (
    <>
      <Hero kicker="Evaluation & evidence" title="Model Insights" description="Explore the evidence behind the deployed model — dataset balance, model selection, cross-validation and holdout performance." />
      <div className="kpi-grid four">
        <KpiCard label="Clean records" value="3,534" note="After missing-value and duplicate cleanup" />
        <KpiCard label="5-Fold CV Macro F1" value="44.24%" note="Primary model-selection metric" />
        <KpiCard label="Holdout Accuracy" value="70.58%" note="Final test size: 707 records" />
        <KpiCard label="Holdout Macro F1" value="45.60%" note="Equal weight across classes" />
      </div>
      <div className="info-box"><b>Why Macro F1?</b> Minimal represents about 73% of the cleaned data, so Accuracy alone can overstate performance for a model that favors the majority class.</div>

      <div className="two-column insights-top">
        <section className="panel-card">
          <h2>Dataset distribution</h2>
          <div className="bar-list">
            {distribution.map((item) => (
              <div className="bar-row" key={item.label}>
                <div className="bar-meta"><b>{item.label}</b><span>{item.count.toLocaleString()} • {item.share.toFixed(2)}%</span></div>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${(item.count / maxDistribution) * 100}%` }} /></div>
              </div>
            ))}
          </div>
        </section>
        <section className="panel-card">
          <h2>Evaluation strategy</h2>
          <Step num="01" title="Development split">Use the development portion for comparison and 5-fold stratified cross-validation.</Step>
          <Step num="02" title="Model selection">Select the method by mean Macro F1 instead of Accuracy alone.</Step>
          <Step num="03" title="Holdout test">Report final performance on the held-out test split.</Step>
        </section>
      </div>

      <section className="panel-card section-card">
        <h2>Model comparison</h2>
        <div className="bar-list">
          {modelComparison.map((item) => (
            <div className="bar-row" key={item.name}>
              <div className="bar-meta"><b>{item.name}</b><span>{item.macroF1.toFixed(2)}% Macro F1</span></div>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${(item.macroF1 / maxF1) * 100}%` }} /></div>
            </div>
          ))}
        </div>
        <div className="info-box compact">The dummy baseline achieves high Accuracy by predicting the dominant class, but its Macro F1 is very low. This is why Accuracy alone is not enough for this dataset.</div>
      </section>

      <div className="two-column confusion-wrap">
        <section className="panel-card">
          <h2>Holdout confusion matrix</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="confusion-image" src="/confusion-matrix.png" alt="Normalized holdout confusion matrix" />
        </section>
        <section className="panel-stack">
          <div className="panel-card">
            <h3>What the matrix tells us</h3>
            <p>Minimal is separated most reliably. Mild remains the hardest class, while Moderate and Severe still show confusion with Minimal.</p>
          </div>
          <div className="panel-card">
            <h3>Selected model</h3>
            <p><b>TF-IDF (Unigram + Bigram)</b><br />Balanced Logistic Regression</p>
            <p>Selected because it provided the strongest Macro F1 among the compared methods while remaining directly explainable through linear coefficients.</p>
          </div>
          <div className="panel-card">
            <h3>Behavioral sanity test</h3>
            <div className="mini-kpis"><b>30</b><span>handcrafted texts</span><b>90%</b><span>marked uncertain</span></div>
            <p className="small-note">This test checks behavior only. It is not Accuracy or clinical validation because the handcrafted examples do not have expert ground truth.</p>
          </div>
        </section>
      </div>
    </>
  );
}

function Step({ num, title, children }: { num: string; title: string; children: React.ReactNode }) {
  return <div className="step"><span>{num}</span><div><b>{title}</b><p>{children}</p></div></div>;
}

function HowItWorks() {
  const pipeline = [
    ["01", "Text input", "The user submits English text only when pressing Analyze."],
    ["02", "TF-IDF features", "Unigrams and bigrams are transformed into a sparse numerical vector."],
    ["03", "Balanced classifier", "Balanced Logistic Regression computes a score for each of the four dataset labels."],
    ["04", "Class scores", "Scores are compared within the model to rank Minimal, Mild, Moderate and Severe."],
    ["05", "Clarity check", "Top score and top-two margin are checked before presenting the result as clear."],
    ["06", "Explanation", "TF-IDF values are combined with learned coefficients to show local feature contribution."],
  ];
  return (
    <>
      <Hero kicker="System design" title="How It Works" description="What happens after a user submits English text — from numerical features to class scores, uncertainty and explainability." />
      <h2>Prediction pipeline</h2>
      <div className="process-grid">
        {pipeline.map(([num, title, text]) => <div className="process-card" key={num}><span>{num}</span><h3>{title}</h3><p>{text}</p></div>)}
      </div>
      <h2>Why this model?</h2>
      <div className="three-column">
        <InfoCard kicker="TF-IDF" title="Strong classical NLP baseline">Suitable for sparse text features and interpretable word/phrase weighting.</InfoCard>
        <InfoCard kicker="Balanced LR" title="Handles class imbalance">`class_weight="balanced"` gives minority classes more training importance without generating synthetic samples.</InfoCard>
        <InfoCard kicker="Explainability" title="Coefficients can be inspected">The linear model lets us inspect how active text features influence a class score.</InfoCard>
      </div>
      <h2>Prediction clarity</h2>
      <div className="code-box">
        <div><b>Uncertain</b><span>top score &lt; 50% OR top-two margin &lt; 10%</span></div>
        <div><b>Relatively clear</b><span>passes the minimum uncertainty rule</span></div>
        <div><b>Clearer prediction</b><span>top score ≥ 65% AND top-two margin ≥ 15%</span></div>
      </div>
      <div className="info-box">These clarity thresholds are demo rules for describing score separation. They are not clinical thresholds and have not been calibrated for medical use.</div>
      <h2>Explainability calculation</h2>
      <div className="formula">Feature contribution<sub>j,c</sub> = TF-IDF<sub>j</sub> × β<sub>j,c</sub></div>
      <p className="section-copy">A positive contribution raises the linear score of that class; a negative contribution lowers it. This explains model mechanics, not medical meaning.</p>
      <h2>What the system does not do</h2>
      <div className="three-column">
        <InfoCard kicker="Boundary" title="No diagnosis">The application does not determine whether a person has depression.</InfoCard>
        <InfoCard kicker="Boundary" title="No clinical risk score">Percentages are model class scores, not clinical probabilities.</InfoCard>
        <InfoCard kicker="Boundary" title="No hidden LLM">The prediction comes from the trained TF-IDF + Logistic Regression pipeline.</InfoCard>
      </div>
    </>
  );
}

function InfoCard({ kicker, title, children }: { kicker: string; title: string; children: React.ReactNode }) {
  return <div className="info-card"><span>{kicker}</span><h3>{title}</h3><p>{children}</p></div>;
}

function AboutProject() {
  return (
    <>
      <Hero kicker="Project profile" title="About the Project" description="An end-to-end NLP portfolio project covering data preparation, model comparison, evaluation, explainability and full-stack deployment." />
      <section className="panel-card section-card">
        <h2>Project overview</h2>
        <p><b>The Use of Artificial Intelligence for Classifying Depression Severity Levels from Text</b></p>
        <p>This system receives English text and classifies it into four labels from the Reddit Depression Dataset: Minimal, Mild, Moderate and Severe. The portfolio version focuses on a complete ML workflow and transparent model explanation.</p>
      </section>
      <h2>What this project demonstrates</h2>
      <div className="four-column">
        <InfoCard kicker="Data" title="Preparation">Duplicate cleanup, class analysis and reproducible training data preparation.</InfoCard>
        <InfoCard kicker="ML" title="Model selection">Baseline creation, method comparison and Macro-F1-based selection.</InfoCard>
        <InfoCard kicker="Eval" title="Evaluation">5-fold CV, holdout testing, confusion matrix and error analysis.</InfoCard>
        <InfoCard kicker="Deploy" title="Full-stack app">Next.js frontend, FastAPI inference API, uncertainty and feature contribution.</InfoCard>
      </div>
      <div className="two-column about-tech">
        <section className="panel-card">
          <h2>Technical profile</h2>
          <div className="chips"><Chip>Python</Chip><Chip>scikit-learn</Chip><Chip>FastAPI</Chip><Chip>Next.js</Chip><Chip>TypeScript</Chip><Chip>Vercel</Chip></div>
          <ul className="plain-list">
            <li><b>Method:</b> TF-IDF Bigram + Balanced Logistic Regression</li>
            <li><b>Training records:</b> 3,534</li>
            <li><b>Vocabulary size:</b> 36,151 features</li>
            <li><b>Language:</b> English</li>
          </ul>
        </section>
        <section className="panel-card">
          <h2>Evaluation highlights</h2>
          <ul className="plain-list">
            <li><b>5-Fold CV Macro F1:</b> 44.24%</li>
            <li><b>Holdout Accuracy:</b> 70.58%</li>
            <li><b>Holdout Macro F1:</b> 45.60%</li>
            <li><b>Selection metric:</b> Macro F1</li>
          </ul>
        </section>
      </div>
      <h2>Known limitations</h2>
      <div className="two-column">
        <InfoCard kicker="Model" title="Data & model limitations">The dataset is imbalanced, Mild is difficult, and TF-IDF cannot deeply understand negation, sarcasm or out-of-domain context.</InfoCard>
        <InfoCard kicker="Responsible use" title="Educational boundary">This portfolio prototype is not a diagnosis, clinical probability, medical recommendation or emergency detection system.</InfoCard>
      </div>
      <div className="success-box">Portfolio v3.0 • Next.js frontend • FastAPI backend • Explainability enabled • Vercel-ready architecture</div>
    </>
  );
}
