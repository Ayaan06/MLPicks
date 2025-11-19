import { useState } from "react";
import {
  fetchPlayerProps,
  type PlayerPropsResponse,
  type StatType,
} from "./api";
import { PropCard } from "./components/PropCard";

type LineState = Record<StatType, string>;

const STAT_TYPES: StatType[] = ["points", "rebounds", "assists"];

const today = new Date().toISOString().split("T")[0];

function App() {
  const [playerId, setPlayerId] = useState("237");
  const [teamId, setTeamId] = useState("14");
  const [opponentTeamId, setOpponentTeamId] = useState("6");
  const [gameDate, setGameDate] = useState(today);
  const [lines, setLines] = useState<LineState>({
    points: "25.5",
    rebounds: "7.5",
    assists: "6.5",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<PlayerPropsResponse | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const propLines = STAT_TYPES.filter(
      (stat) => lines[stat] !== "" && !Number.isNaN(Number(lines[stat])),
    ).map((stat) => ({
      stat_type: stat,
      line: Number(lines[stat]),
    }));

    if (propLines.length === 0) {
      setError("Please enter at least one prop line.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await fetchPlayerProps({
        player_id: playerId,
        team_id: teamId,
        opponent_team_id: opponentTeamId,
        game_date: gameDate,
        prop_lines: propLines,
      });
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch picks.");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>ML Picks – Player Props</h1>
        <p>Enter a matchup and prop lines to get ML-driven picks.</p>
      </header>
      <section className="form-section">
        <form onSubmit={handleSubmit} className="props-form">
          <div className="form-grid">
            <label>
              Player ID
              <input
                value={playerId}
                onChange={(e) => setPlayerId(e.target.value)}
                required
              />
            </label>
            <label>
              Team ID
              <input
                value={teamId}
                onChange={(e) => setTeamId(e.target.value)}
                required
              />
            </label>
            <label>
              Opponent Team ID
              <input
                value={opponentTeamId}
                onChange={(e) => setOpponentTeamId(e.target.value)}
                required
              />
            </label>
            <label>
              Game Date
              <input
                type="date"
                value={gameDate}
                onChange={(e) => setGameDate(e.target.value)}
                required
              />
            </label>
          </div>
          <div className="line-inputs">
            {STAT_TYPES.map((stat) => (
              <label key={stat}>
                {stat.charAt(0).toUpperCase() + stat.slice(1)} Line
                <input
                  type="number"
                  step="0.5"
                  value={lines[stat]}
                  onChange={(e) =>
                    setLines((prev) => ({
                      ...prev,
                      [stat]: e.target.value,
                    }))
                  }
                />
              </label>
            ))}
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Loading..." : "Get Picks"}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
      </section>
      <section className="results-section">
        {response ? (
          <div className="cards-grid">
            {STAT_TYPES.map((stat) => {
              const prediction = response.props[stat];
              return (
                prediction && (
                  <PropCard key={stat} prediction={prediction} />
                )
              );
            })}
          </div>
        ) : (
          <p className="placeholder">Results will appear here.</p>
        )}
      </section>
    </div>
  );
}

export default App;
