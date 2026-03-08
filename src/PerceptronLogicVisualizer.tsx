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

  return (
    <div style={{ ...cardStyle(), height: "100%" }}>
      <h2 style={sectionTitleStyle()}>퍼셉트론 구조</h2>
      <svg viewBox="0 0 520 240" style={{ width: "100%", height: "240px" }}>
        <circle cx="90" cy="70" r="28" fill="#ffffff" stroke="#334155" strokeWidth="2" />
        <circle cx="90" cy="150" r="28" fill="#ffffff" stroke="#334155" strokeWidth="2" />

        <circle cx="270" cy="110" r="52" fill="#f8fafc" stroke="#334155" strokeWidth="2.5" />
        <rect x="340" y="72" width="56" height="76" rx="14" fill="#dbeafe" stroke="#2563eb" strokeWidth="2" />

        <line x1="118" y1="70" x2="220" y2="90" stroke="#334155" strokeWidth="2" />
        <line x1="118" y1="150" x2="220" y2="130" stroke="#334155" strokeWidth="2" />
        <line x1="322" y1="110" x2="340" y2="110" stroke="#334155" strokeWidth="2.5" />
        <line x1="396" y1="110" x2="430" y2="110" stroke="#334155" strokeWidth="2.5" />
        <polygon points="430,110 416,102 416,118" fill="#334155" />

        <text x="82" y="76" fontSize="18" fill="#0f172a">x₁</text>
        <text x="82" y="156" fontSize="18" fill="#0f172a">x₂</text>

        <text x="42" y="38" fontSize="14" fill="#475569">입력값</text>
        <text x="236" y="36" fontSize="14" fill="#475569">가중합 s</text>
        <text x="342" y="36" fontSize="14" fill="#475569">활성화 f</text>
        <text x="444" y="88" fontSize="14" fill="#475569">출력</text>

        <text x="257" y="116" fontSize="22" fill="#0f172a">Σ</text>
        <text x="360" y="118" fontSize="28" fill="#1e3a8a">f</text>
        <text x="448" y="116" fontSize="18" fill="#0f172a">y = {output}</text>
        <text x="327" y="64" fontSize="13" fill="#0f172a">임계값 θ = {theta}</text>
        <text x="322" y="164" fontSize="12" fill="#475569">y = 1 (s ≥ θ), 0 (s &lt; θ)</text>

        <text x="145" y="72" fontSize="14" fill="#0f172a">w₁ = {w1}</text>
        <text x="145" y="150" fontSize="14" fill="#0f172a">w₂ = {w2}</text>

        <text x="50" y="102" fontSize="14" fill="#0f172a">{x1}</text>
        <text x="50" y="182" fontSize="14" fill="#0f172a">{x2}</text>

        <text x="156" y="190" fontSize="13" fill="#0f172a">
          s = x₁w₁ + x₂w₂ = {x1}×{w1} + {x2}×{w2} = {weightedSum}
        </text>
        <text x="156" y="208" fontSize="13" fill="#0f172a">
          s - θ = {weightedSum} - {theta} = {activationInput} → y = {output}
        </text>
      </svg>
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
  const [w1, setW1] = useState<number>(0.5);
  const [w2, setW2] = useState<number>(0.5);
  const [bias, setBias] = useState<number>(-0.7);
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

    if (nextLogic === "AND") {
      setW1(0.5);
      setW2(0.5);
      setBias(-0.7);
    } else if (nextLogic === "OR") {
      setW1(0.6);
      setW2(0.6);
      setBias(-0.3);
    } else if (nextLogic === "NAND") {
      setW1(-0.6);
      setW2(-0.6);
      setBias(0.9);
    } else {
      setW1(0.2);
      setW2(-0.1);
      setBias(0);
    }
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

            <div
              style={{
                background: "#eff6ff",
                border: "1px solid #bfdbfe",
                borderRadius: "16px",
                padding: "14px",
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: "8px" }}>현재 상태</div>
              <div style={{ marginBottom: "4px" }}>정확도: {accuracy}%</div>
              <div style={{ marginBottom: "4px" }}>학습 스텝: {learningStep}</div>
              <div style={{ marginBottom: "4px" }}>임계값 θ: {theta}</div>
              <div style={{ marginBottom: "4px" }}>선택한 입력: ({sample.x1}, {sample.x2})</div>
              <div style={{ marginBottom: "4px" }}>정답: {sample.y}</div>
              <div>자동 학습: {autoLearning ? "실행 중" : "정지"}</div>
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
                <div style={{ overflowX: "auto", maxHeight: "320px", overflowY: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      minWidth: "980px",
                    }}
                  >
                    <thead>
                      <tr style={{ background: "#f8fafc" }}>
                        <th style={tableHeaderStyle}>step</th>
                        <th style={tableHeaderStyle}>샘플</th>
                        <th style={tableHeaderStyle}>정답/예측</th>
                        <th style={tableHeaderStyle}>오차</th>
                        <th style={tableHeaderStyle}>w1</th>
                        <th style={tableHeaderStyle}>w2</th>
                        <th style={tableHeaderStyle}>θ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {learningLogs.map((log) => (
                        <tr key={log.id}>
                          <td style={tableCellStyle}>{log.step}</td>
                          <td style={tableCellStyle}>
                            #{log.sampleIndex + 1} ({log.x1}, {log.x2})
                          </td>
                          <td style={tableCellStyle}>
                            {log.target} / {log.pred}
                          </td>
                          <td style={tableCellStyle}>{log.error}</td>
                          <td style={tableCellStyle}>
                            {log.w1Before} → {log.w1After}
                          </td>
                          <td style={tableCellStyle}>
                            {log.w2Before} → {log.w2After}
                          </td>
                          <td style={tableCellStyle}>
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
