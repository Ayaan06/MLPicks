import type { SinglePropPrediction, StatType } from "../api";

const confidenceColor: Record<
  SinglePropPrediction["confidence_label"],
  string
> = {
  low: "#9CA3AF",
  medium: "#FACC15",
  high: "#86EFAC",
  very_high: "#16A34A",
};

const riskColor: Record<SinglePropPrediction["risk_label"], string> = {
  low: "#16A34A",
  medium: "#FACC15",
  high: "#EF4444",
};

function titleCase(stat: StatType): string {
  return stat.charAt(0).toUpperCase() + stat.slice(1);
}

interface PropCardProps {
  prediction: SinglePropPrediction;
}

export function PropCard({ prediction }: PropCardProps) {
  const {
    stat_type,
    model_projection,
    line,
    pick_side,
    confidence_score,
    confidence_label,
    edge_value,
    edge_prob,
    risk_label,
    reason,
  } = prediction;

  const confidenceWidth = `${Math.round(confidence_score)}%`;

  return (
    <div className="prop-card">
      <div className="prop-card__header">
        <h3>{titleCase(stat_type)}</h3>
        <span className="prop-card__pick">{pick_side.toUpperCase()}</span>
      </div>
      <div className="prop-card__row">
        <span>Model Projection</span>
        <strong>{model_projection.toFixed(1)}</strong>
      </div>
      <div className="prop-card__row">
        <span>Line</span>
        <strong>{line.toFixed(1)}</strong>
      </div>
      <div className="prop-card__row">
        <span>Confidence</span>
        <strong>{confidence_score.toFixed(1)}%</strong>
      </div>
      <div className="confidence-bar">
        <div
          className="confidence-bar__fill"
          style={{
            width: confidenceWidth,
            backgroundColor: confidenceColor[confidence_label],
          }}
        />
      </div>
      <div className="prop-card__meta">
        <span
          className="label"
          style={{ backgroundColor: confidenceColor[confidence_label] }}
        >
          {confidence_label.replace("_", " ")}
        </span>
        <span className="edge">
          Edge: {edge_value.toFixed(2)} ({(edge_prob * 100).toFixed(1)} pp)
        </span>
        <span
          className="label"
          style={{ backgroundColor: riskColor[risk_label] }}
        >
          {risk_label} risk
        </span>
      </div>
      {reason && <p className="prop-card__reason">{reason}</p>}
    </div>
  );
}
