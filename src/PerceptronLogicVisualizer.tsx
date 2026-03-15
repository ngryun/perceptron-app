import { useCallback, useEffect, useMemo, useState } from "react";

type LogicType = "AND" | "OR" | "NAND" | "XOR";

type Sample = {
  x1: number;
  x2: number;
  y: number;
};

type LearningLog = {
  id: string;
  step: number;
  sampleIndex: number;
  x1: number;
  x2: number;
  target: number;
  pred: number;
  error: number;
  w1Before: number;
  w2Before: number;
  thetaBefore: number;
  w1After: number;
  w2After: number;
  thetaAfter: number;
};

const DATASETS: Record<LogicType, Sample[]> = {
  AND: [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 0 },
    { x1: 1, x2: 0, y: 0 },
    { x1: 1, x2: 1, y: 1 },
  ],
  OR: [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 1 },
  ],
  NAND: [
    { x1: 0, x2: 0, y: 1 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 0 },
  ],
  XOR: [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 0 },
  ],
};

const LEARNING_RATE = 0.1;
const MAX_LOG_COUNT = 120;

function step(z: number): number {
  return z >= 0 ? 1 : 0;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function randomWeight(): number {
  return Math.round((Math.random() * 2 - 1) * 10) / 10;
}

function toTheta(bias: number): number {
  return round3(-bias);
}

function compute(sample: Sample, w1: number, w2: number, bias: number) {
  const net = sample.x1 * w1 + sample.x2 * w2 + bias;
  const output = step(net);
  return { net, output };
}

function cardStyle(): React.CSSProperties {
  return {
    background: "#ffffff",
    border: "1px solid #dbe3ea",
    borderRadius: "20px",
    padding: "20px",
    boxShadow: "0 8px 24px rgba(15, 23, 42, 0.06)",
    boxSizing: "border-box",
    minWidth: 0,
  };
}

function sectionTitleStyle(): React.CSSProperties {
  return {
    margin: "0 0 12px 0",
    fontSize: "20px",
    fontWeight: 700,
    color: "#0f172a",
  };
}

function buttonStyle(active: boolean): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderRadius: "12px",
    border: active ? "1px solid #2563eb" : "1px solid #cbd5e1",
    background: active ? "#dbeafe" : "#ffffff",
    color: "#0f172a",
    cursor: "pointer",
    fontWeight: 600,
  };
}

function sliderRowStyle(): React.CSSProperties {
  return {
    display: "grid",
    gridTemplateColumns: "70px 1fr 60px",
    gap: "12px",
    alignItems: "center",
    marginBottom: "14px",
  };
}

function PerceptronDiagram(props: {
  x1: number;
  x2: number;
  w1: number;
  w2: number;
  bias: number;
  net: number;
  output: number;
}) {
  const { x1, x2, w1, w2, bias, net, output } = props;
  const theta = toTheta(bias);
  const weightedSum = round3(x1 * w1 + x2 * w2);
  const activationInput = round3(net);
  const summaryBoxStyle: React.CSSProperties = {
    background: "#f8fafc",
    border: "1px solid #dbeafe",
    borderRadius: "16px",
    padding: "18px 20px",
    boxSizing: "border-box",
  };

  return (
    <div style={{ ...cardStyle(), height: "100%" }}>
      <h2 style={sectionTitleStyle()}>퍼셉트론 구조</h2>
      <svg
        viewBox="0 0 760 360"
        style={{ width: "100%", height: "auto", display: "block", margin: "0 auto" }}
      >
        <defs>
          <marker
            id="perceptron-arrow"
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="5"
            orient="auto"
          >
            <path d="M0,0 L10,5 L0,10 Z" fill="#334155" />
          </marker>
        </defs>

        <text x="70" y="50" fontSize="20" fontWeight="700" fill="#475569">입력값</text>
        <text x="294" y="50" fontSize="20" fontWeight="700" fill="#475569">가중합</text>
        <text x="484" y="50" fontSize="20" fontWeight="700" fill="#475569">판정</text>
        <text x="656" y="50" fontSize="20" fontWeight="700" fill="#475569">출력</text>

        <circle cx="96" cy="124" r="46" fill="#ffffff" stroke="#334155" strokeWidth="4" />
        <circle cx="96" cy="254" r="46" fill="#ffffff" stroke="#334155" strokeWidth="4" />
        <text x="82" y="116" fontSize="32" fontWeight="700" fill="#0f172a">x₁</text>
        <text x="82" y="246" fontSize="32" fontWeight="700" fill="#0f172a">x₂</text>
        <text x="89" y="146" fontSize="30" fontWeight="700" fill="#2563eb">{x1}</text>
        <text x="89" y="276" fontSize="30" fontWeight="700" fill="#2563eb">{x2}</text>

        <line x1="142" y1="124" x2="272" y2="164" stroke="#334155" strokeWidth="5" markerEnd="url(#perceptron-arrow)" />
        <line x1="142" y1="254" x2="272" y2="204" stroke="#334155" strokeWidth="5" markerEnd="url(#perceptron-arrow)" />
        <text x="176" y="118" fontSize="24" fontWeight="700" fill="#0f172a">w₁ = {round3(w1)}</text>
        <text x="176" y="246" fontSize="24" fontWeight="700" fill="#0f172a">w₂ = {round3(w2)}</text>

        <circle cx="360" cy="184" r="84" fill="#f8fafc" stroke="#334155" strokeWidth="5" />
        <text x="344" y="172" fontSize="46" fontWeight="700" fill="#0f172a">Σ</text>
        <text x="308" y="212" fontSize="26" fontWeight="700" fill="#475569">s = {weightedSum}</text>

        <line x1="444" y1="184" x2="500" y2="184" stroke="#334155" strokeWidth="5" markerEnd="url(#perceptron-arrow)" />

        <rect x="520" y="118" width="126" height="134" rx="28" fill="#dbeafe" stroke="#2563eb" strokeWidth="5" />
        <text x="572" y="170" fontSize="34" fontWeight="700" fill="#1e3a8a">f</text>
        <text x="547" y="206" fontSize="23" fontWeight="700" fill="#1e3a8a">θ = {theta}</text>
        <text x="534" y="232" fontSize="17" fontWeight="700" fill="#1e3a8a">s와 θ를 비교</text>

        <line x1="646" y1="184" x2="680" y2="184" stroke="#334155" strokeWidth="5" markerEnd="url(#perceptron-arrow)" />

        <rect
          x="684"
          y="136"
          width="58"
          height="96"
          rx="22"
          fill={output === 1 ? "#dcfce7" : "#fee2e2"}
          stroke={output === 1 ? "#15803d" : "#b91c1c"}
          strokeWidth="5"
        />
        <text x="705" y="196" fontSize="38" fontWeight="700" fill="#0f172a">{output}</text>
      </svg>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: "14px",
          marginTop: "18px",
        }}
      >
        <div style={summaryBoxStyle}>
          <div style={{ color: "#475569", fontSize: "15px", fontWeight: 700, marginBottom: "8px" }}>
            1. 입력과 가중치
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            x₁ = {x1}, x₂ = {x2}
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            w₁ = {round3(w1)}, w₂ = {round3(w2)}
          </div>
        </div>

        <div style={summaryBoxStyle}>
          <div style={{ color: "#475569", fontSize: "15px", fontWeight: 700, marginBottom: "8px" }}>
            2. 가중합 계산
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            s = x₁w₁ + x₂w₂
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            = {x1}×{round3(w1)} + {x2}×{round3(w2)}
          </div>
          <div style={{ color: "#2563eb", fontSize: "24px", fontWeight: 700, lineHeight: 1.4 }}>
            s = {weightedSum}
          </div>
        </div>

        <div style={summaryBoxStyle}>
          <div style={{ color: "#475569", fontSize: "15px", fontWeight: 700, marginBottom: "8px" }}>
            3. 임계값과 비교
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            θ = {theta}
          </div>
          <div style={{ color: "#0f172a", fontSize: "18px", lineHeight: 1.6 }}>
            s - θ = {weightedSum} - {theta}
          </div>
          <div style={{ color: "#0f172a", fontSize: "24px", fontWeight: 700, lineHeight: 1.4 }}>
            {activationInput} {activationInput >= 0 ? "≥" : "<"} 0 → y = {output}
          </div>
        </div>
      </div>
    </div>
  );
}

function BoundaryChart(props: {
  dataset: Sample[];
  w1: number;
  w2: number;
  bias: number;
  selectedIndex: number;
}) {
  const { dataset, w1, w2, bias, selectedIndex } = props;
  const size = 320;
  const pad = 40;
  const min = -0.2;
  const max = 1.2;

  function mapX(x: number) {
    return pad + ((x - min) / (max - min)) * (size - pad * 2);
  }

  function mapY(y: number) {
    return size - pad - ((y - min) / (max - min)) * (size - pad * 2);
  }

  let line: { x1: number; y1: number; x2: number; y2: number } | null = null;

  if (Math.abs(w1) < 0.0001 && Math.abs(w2) < 0.0001) {
    line = null;
  } else if (Math.abs(w2) < 0.0001) {
    const x = -bias / w1;
    line = { x1: x, y1: min, x2: x, y2: max };
  } else {
    const yAtMin = (-bias - w1 * min) / w2;
    const yAtMax = (-bias - w1 * max) / w2;
    line = { x1: min, y1: yAtMin, x2: max, y2: yAtMax };
  }

  return (
    <div style={{ ...cardStyle(), height: "100%" }}>
      <h2 style={sectionTitleStyle()}>결정경계 그래프</h2>

      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxWidth: "340px", display: "block", margin: "0 auto" }}>
        <rect x="0" y="0" width={size} height={size} rx="18" fill="#ffffff" />
        <line x1={pad} y1={size - pad} x2={size - pad} y2={size - pad} stroke="#334155" strokeWidth="2" />
        <line x1={pad} y1={size - pad} x2={pad} y2={pad} stroke="#334155" strokeWidth="2" />

        <text x={size - pad + 5} y={size - pad + 5} fontSize="12" fill="#0f172a">x₁</text>
        <text x={pad - 14} y={pad - 10} fontSize="12" fill="#0f172a">x₂</text>

        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line x1={mapX(tick)} y1={size - pad} x2={mapX(tick)} y2={size - pad + 5} stroke="#334155" />
            <text x={mapX(tick) - 8} y={size - pad + 18} fontSize="11" fill="#475569">{tick}</text>
            <line x1={pad - 5} y1={mapY(tick)} x2={pad} y2={mapY(tick)} stroke="#334155" />
            <text x={pad - 26} y={mapY(tick) + 4} fontSize="11" fill="#475569">{tick}</text>
          </g>
        ))}

        {line && (
          <line
            x1={mapX(line.x1)}
            y1={mapY(line.y1)}
            x2={mapX(line.x2)}
            y2={mapY(line.y2)}
            stroke="#2563eb"
            strokeWidth="3"
          />
        )}

        {dataset.map((point, index) => {
          const active = index === selectedIndex;
          const fill = point.y === 1 ? "#ffffff" : "#e2e8f0";

          return (
            <g key={`${point.x1}-${point.x2}-${index}`}>
              <circle
                cx={mapX(point.x1)}
                cy={mapY(point.x2)}
                r={active ? 13 : 10}
                fill={fill}
                stroke={active ? "#2563eb" : "#334155"}
                strokeWidth={active ? 3 : 2}
              />
              <text
                x={mapX(point.x1) - 4}
                y={mapY(point.x2) + 4}
                fontSize="11"
                fill="#0f172a"
              >
                {point.y}
              </text>
            </g>
          );
        })}
      </svg>

      <p style={{ marginTop: "14px", color: "#475569", fontSize: "14px" }}>
        파란 직선이 현재 결정경계입니다. 가중치와 임계값(θ)을 바꾸면 선의 위치와 기울기가 함께 바뀝니다.
      </p>
    </div>
  );
}

export default function PerceptronLogicVisualizer() {
  const [logicType, setLogicType] = useState<LogicType>("AND");
  const [w1, setW1] = useState<number>(() => randomWeight());
  const [w2, setW2] = useState<number>(() => randomWeight());
  const [bias, setBias] = useState<number>(() => randomWeight());
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [trainCursor, setTrainCursor] = useState<number>(0);
  const [learningStep, setLearningStep] = useState<number>(0);
  const [autoLearning, setAutoLearning] = useState<boolean>(false);
  const [learningLogs, setLearningLogs] = useState<LearningLog[]>([]);

  const dataset = DATASETS[logicType];
  const sample = dataset[selectedIndex];
  const theta = toTheta(bias);

  const result = useMemo(() => {
    return compute(sample, w1, w2, bias);
  }, [sample, w1, w2, bias]);

  const rows = useMemo(() => {
    return dataset.map((item) => {
      const value = compute(item, w1, w2, bias);
      return {
        ...item,
        net: round(value.net),
        pred: value.output,
        correct: value.output === item.y,
      };
    });
  }, [dataset, w1, w2, bias]);

  const accuracy = useMemo(() => {
    const correctCount = rows.filter((row) => row.correct).length;
    return Math.round((correctCount / rows.length) * 100);
  }, [rows]);

  const trainOneStep = useCallback(() => {
    const trainSample = dataset[trainCursor];
    const value = compute(trainSample, w1, w2, bias);
    const error = trainSample.y - value.output;
    const nextW1 = round3(w1 + LEARNING_RATE * error * trainSample.x1);
    const nextW2 = round3(w2 + LEARNING_RATE * error * trainSample.x2);
    const nextBias = round3(bias + LEARNING_RATE * error);
    const nextStep = learningStep + 1;

    setW1(nextW1);
    setW2(nextW2);
    setBias(nextBias);
    setSelectedIndex(trainCursor);
    setTrainCursor((prev) => (prev + 1) % dataset.length);
    setLearningStep(nextStep);
    setLearningLogs((prev) => [
      {
        id: `${nextStep}-${trainCursor}-${Date.now()}`,
        step: nextStep,
        sampleIndex: trainCursor,
        x1: trainSample.x1,
        x2: trainSample.x2,
        target: trainSample.y,
        pred: value.output,
        error,
        w1Before: round3(w1),
        w2Before: round3(w2),
        thetaBefore: toTheta(bias),
        w1After: nextW1,
        w2After: nextW2,
        thetaAfter: toTheta(nextBias),
      },
      ...prev,
    ].slice(0, MAX_LOG_COUNT));
  }, [dataset, trainCursor, w1, w2, bias, learningStep]);

  useEffect(() => {
    if (!autoLearning) {
      return;
    }

    const intervalId = window.setInterval(() => {
      trainOneStep();
    }, 700);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoLearning, trainOneStep]);

  useEffect(() => {
    if (autoLearning && logicType !== "XOR" && accuracy === 100) {
      setAutoLearning(false);
    }
  }, [autoLearning, logicType, accuracy]);

  function resetPreset(nextLogic: LogicType) {
    setLogicType(nextLogic);
    setSelectedIndex(0);
    setTrainCursor(0);
    setLearningStep(0);
    setAutoLearning(false);
    setLearningLogs([]);

    setW1(randomWeight());
    setW2(randomWeight());
    setBias(randomWeight());
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%)",
        color: "#0f172a",
        padding: "24px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ maxWidth: "1500px", margin: "0 auto" }}>
        <div
          style={{
            ...cardStyle(),
            marginBottom: "20px",
            background: "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)",
            color: "#ffffff",
          }}
        >
          <h1 style={{ margin: 0, fontSize: "34px" }}>퍼셉트론 논리연산 시각화</h1>
          <p style={{ marginTop: "10px", marginBottom: 0, fontSize: "16px", color: "#dbeafe" }}>
            입력값, 가중치, 임계값(θ), 출력, 진리표, 결정경계를 한 화면에서 확인할 수 있습니다.
          </p>
        </div>

        <div className="layout-top-grid">
          <div style={cardStyle()}>
            <h2 style={sectionTitleStyle()}>제어 패널</h2>

            <div style={{ marginBottom: "18px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: 700 }}>
                논리연산
              </label>
              <select
                value={logicType}
                onChange={(e) => resetPreset(e.target.value as LogicType)}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: "12px",
                  border: "1px solid #cbd5e1",
                  fontSize: "15px",
                  color: "#0f172a",
                  background: "#ffffff",
                }}
              >
                <option value="AND">AND</option>
                <option value="OR">OR</option>
                <option value="NAND">NAND</option>
                <option value="XOR">XOR</option>
              </select>
            </div>

            <div style={{ marginBottom: "18px" }}>
              <div style={sliderRowStyle()}>
                <label style={{ fontWeight: 700 }}>w1</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={w1}
                  onChange={(e) => setW1(Number(e.target.value))}
                />
                <span>{w1}</span>
              </div>

              <div style={sliderRowStyle()}>
                <label style={{ fontWeight: 700 }}>w2</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={w2}
                  onChange={(e) => setW2(Number(e.target.value))}
                />
                <span>{w2}</span>
              </div>

              <div style={sliderRowStyle()}>
                <label style={{ fontWeight: 700 }}>임계값 θ</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={theta}
                  onChange={(e) => setBias(-Number(e.target.value))}
                />
                <span>{theta}</span>
              </div>
            </div>

            <div style={{ marginBottom: "18px" }}>
              <div style={{ marginBottom: "10px", fontWeight: 700 }}>입력 선택</div>
              <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                {dataset.map((item, index) => (
                  <button
                    key={index}
                    onClick={() => setSelectedIndex(index)}
                    style={buttonStyle(selectedIndex === index)}
                  >
                    ({item.x1}, {item.x2})
                  </button>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: "18px" }}>
              <div style={{ marginBottom: "10px", fontWeight: 700 }}>학습 제어</div>
              <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                <button onClick={trainOneStep} style={buttonStyle(false)}>
                  한 단계 학습
                </button>
                <button onClick={() => setAutoLearning((prev) => !prev)} style={buttonStyle(autoLearning)}>
                  {autoLearning ? "자동 학습 중지" : "자동 학습 시작"}
                </button>
                <button onClick={() => setLearningLogs([])} style={buttonStyle(false)}>
                  로그 지우기
                </button>
              </div>
              <div style={{ marginTop: "10px", color: "#475569", fontSize: "14px", lineHeight: 1.5 }}>
                <div>학습률 η = {LEARNING_RATE}</div>
                <div>
                  다음 학습 샘플: ({dataset[trainCursor].x1}, {dataset[trainCursor].x2}) / 정답 {dataset[trainCursor].y}
                </div>
              </div>
            </div>
          </div>

          <div style={{ minWidth: 0 }}>
            <PerceptronDiagram
              x1={sample.x1}
              x2={sample.x2}
              w1={w1}
              w2={w2}
              bias={bias}
              net={result.net}
              output={result.output}
            />
          </div>

          <div className="layout-boundary" style={{ minWidth: 0 }}>
            <BoundaryChart
              dataset={dataset}
              w1={w1}
              w2={w2}
              bias={bias}
              selectedIndex={selectedIndex}
            />
          </div>
        </div>

        {logicType === "XOR" && (
          <div
            style={{
              ...cardStyle(),
              marginTop: "20px",
              background: "#fff1f2",
              border: "1px solid #fecdd3",
              color: "#9f1239",
              lineHeight: 1.6,
            }}
          >
            XOR는 단층 퍼셉트론으로 완전히 분리하기 어렵습니다. 점들이 직선 하나로 깔끔하게 나뉘지 않기 때문입니다.
          </div>
        )}

        <div className="layout-bottom-grid">
          <div style={cardStyle()}>
            <h2 style={sectionTitleStyle()}>진리표와 예측</h2>

            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  minWidth: "700px",
                }}
              >
                <thead>
                  <tr style={{ background: "#f8fafc" }}>
                    <th style={tableHeaderStyle}>x1</th>
                    <th style={tableHeaderStyle}>x2</th>
                    <th style={tableHeaderStyle}>정답</th>
                    <th style={tableHeaderStyle}>net</th>
                    <th style={tableHeaderStyle}>예측</th>
                    <th style={tableHeaderStyle}>판정</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr
                      key={index}
                      style={{
                        background:
                          selectedIndex === index ? "#eff6ff" : "#ffffff",
                      }}
                    >
                      <td style={tableCellStyle}>{row.x1}</td>
                      <td style={tableCellStyle}>{row.x2}</td>
                      <td style={tableCellStyle}>{row.y}</td>
                      <td style={tableCellStyle}>{row.net}</td>
                      <td style={tableCellStyle}>{row.pred}</td>
                      <td style={tableCellStyle}>
                        <span
                          style={{
                            display: "inline-block",
                            padding: "6px 10px",
                            borderRadius: "999px",
                            fontSize: "13px",
                            fontWeight: 700,
                            background: row.correct ? "#dcfce7" : "#fee2e2",
                            color: row.correct ? "#166534" : "#991b1b",
                          }}
                        >
                          {row.correct ? "정답" : "오답"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={cardStyle()}>
            <h2 style={sectionTitleStyle()}>학습 로그</h2>

            <div
              style={{
                border: "1px solid #e2e8f0",
                borderRadius: "12px",
                overflow: "hidden",
              }}
            >
              {learningLogs.length === 0 ? (
                <div style={{ padding: "16px", color: "#64748b" }}>
                  아직 학습 로그가 없습니다. "한 단계 학습" 또는 "자동 학습 시작"을 눌러주세요.
                </div>
              ) : (
                <div style={{ overflowX: "hidden" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      tableLayout: "fixed",
                    }}
                  >
                    <thead>
                      <tr style={{ background: "#f8fafc" }}>
                        <th style={logTableHeaderStyle}>step</th>
                        <th style={logTableHeaderStyle}>샘플</th>
                        <th style={logTableHeaderStyle}>정답/예측</th>
                        <th style={logTableHeaderStyle}>오차</th>
                        <th style={logTableHeaderStyle}>w1</th>
                        <th style={logTableHeaderStyle}>w2</th>
                        <th style={logTableHeaderStyle}>θ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {learningLogs.map((log) => (
                        <tr key={log.id}>
                          <td style={logTableCellStyle}>{log.step}</td>
                          <td style={logTableCellStyle}>
                            #{log.sampleIndex + 1} ({log.x1}, {log.x2})
                          </td>
                          <td style={logTableCellStyle}>
                            {log.target} / {log.pred}
                          </td>
                          <td style={logTableCellStyle}>{log.error}</td>
                          <td style={logTableCellStyle}>
                            {log.w1Before} → {log.w1After}
                          </td>
                          <td style={logTableCellStyle}>
                            {log.w2Before} → {log.w2After}
                          </td>
                          <td style={logTableCellStyle}>
                            {log.thetaBefore} → {log.thetaAfter}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: "24px",
            padding: "0 4px 8px",
            color: "#64748b",
            fontSize: "13px",
            lineHeight: 1.7,
            textAlign: "center",
          }}
        >
          <div>Copyright © Namgungyeon. All rights reserved.</div>
          <div>
            수업 소개 및 활용 글:
            {" "}
            <a
              href="https://namgungyeon.tistory.com/154"
              target="_blank"
              rel="noreferrer"
              style={{
                color: "#1d4ed8",
                fontWeight: 600,
                textDecoration: "none",
              }}
            >
              https://namgungyeon.tistory.com/154
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

const tableHeaderStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "12px",
  borderBottom: "1px solid #e2e8f0",
  color: "#0f172a",
};

const tableCellStyle: React.CSSProperties = {
  padding: "12px",
  borderBottom: "1px solid #e2e8f0",
  color: "#0f172a",
};

const logTableHeaderStyle: React.CSSProperties = {
  ...tableHeaderStyle,
  padding: "8px 10px",
  fontSize: "13px",
  whiteSpace: "nowrap",
};

const logTableCellStyle: React.CSSProperties = {
  ...tableCellStyle,
  padding: "7px 10px",
  fontSize: "13px",
  lineHeight: 1.35,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
};
