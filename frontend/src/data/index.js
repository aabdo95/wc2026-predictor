// Pre-baked simulation artefacts, bundled at build time so the deployed
// dashboard needs no backend at runtime.
//
// Raw artefacts (copied verbatim from data/processed/) are transformed
// client-side by useStaticData.js to mirror the FastAPI responses. The three
// "derived" files (overview / model_info / features) are baked snapshots of
// the exact endpoint output, because their values come from the trained
// ensemble.joblib pickle and cannot be recomputed in the browser.

import groupStandings from './group_standings.json'
import matchPredictions from './match_predictions.json'
import winnerOdds from './tournament_winner_odds.json'
import bracket from './bracket.json'
import knockoutProbabilities from './knockout_probabilities.json'
import matchExplanations from './match_explanations.json'

// Baked endpoint snapshots (exact API output)
import overview from './overview.json'
import modelInfo from './model_info.json'
import features from './features.json'

export {
  groupStandings,
  matchPredictions,
  winnerOdds,
  bracket,
  knockoutProbabilities,
  matchExplanations,
  overview,
  modelInfo,
  features,
}
