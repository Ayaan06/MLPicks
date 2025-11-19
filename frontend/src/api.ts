export type StatType = "points" | "rebounds" | "assists";

export interface PlayerPropLine {
  stat_type: StatType;
  line: number;
}

export interface PlayerPropsRequest {
  player_id: string;
  team_id: string;
  opponent_team_id: string;
  game_date: string;
  prop_lines: PlayerPropLine[];
}

export interface SinglePropPrediction {
  stat_type: StatType;
  model_projection: number;
  line: number;
  prob_over: number;
  prob_under: number;
  pick_side: "over" | "under";
  confidence_score: number;
  confidence_label: "low" | "medium" | "high" | "very_high";
  edge_value: number;
  edge_prob: number;
  risk_score: number;
  risk_label: "low" | "medium" | "high";
  reason?: string | null;
}

export interface PlayerPropsResponse {
  player_id: string;
  team_id: string;
  opponent_team_id: string;
  game_date: string;
  props: Record<StatType, SinglePropPrediction>;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchPlayerProps(
  payload: PlayerPropsRequest,
): Promise<PlayerPropsResponse> {
  const response = await fetch(`${API_URL}/player_props`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to fetch picks");
  }
  return response.json();
}
