from math import pi
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "pokemon.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ML_TEST_SIZE = 0.30
ML_ARTIFACT_PATH = ARTIFACTS_DIR / "ml_results_30.pkl"

SECTION_DATA = "Data Preview"
SECTION_DUAL = "Analysis of Pokémon Dual Types"
SECTION_PROFILE = "Analysis of Pokémon Personality Profiles"
SECTION_CAPTURE = "Exploratory Analysis of Pokémon Capture Difficulty"
SECTION_ML = "Machine Learning"

NAV_SECTIONS = [SECTION_DATA, SECTION_DUAL, SECTION_PROFILE, SECTION_CAPTURE, SECTION_ML]

CHART_HEIGHT_FULL = 640
CHART_HEIGHT_HALF = 480
CARD_HEIGHT_FULL = 700
CARD_HEIGHT_HALF = 540

# Default type subset for readable multi-series charts (not all 18 at once).
DEFAULT_SHOWCASE_TYPES = ["fire", "water", "grass"]

TYPE_COLORS = {
    "normal": "#7C7C61",
    "fire": "#C7602B",
    "water": "#4B70B8",
    "electric": "#C89C24",
    "grass": "#5B9B49",
    "ice": "#6FA6A6",
    "fighting": "#9C332E",
    "poison": "#7E3F7F",
    "ground": "#B39552",
    "flying": "#7D6CB8",
    "psychic": "#BB4C73",
    "bug": "#7E8A2C",
    "rock": "#8C7B35",
    "ghost": "#594772",
    "dragon": "#5A45A5",
    "dark": "#5E5148",
    "steel": "#7F8396",
    "fairy": "#B66F84",
}

STAT_COLUMNS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
PROFILE_LABELS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
STAT_MAP = {
    "hp": "HP",
    "attack": "Atk",
    "defense": "Def",
    "sp_attack": "SpA",
    "sp_defense": "SpD",
    "speed": "Spe",
}
RADAR_ANGLES = [n / len(PROFILE_LABELS) * 2 * pi for n in range(len(PROFILE_LABELS))]
RADAR_ANGLES += RADAR_ANGLES[:1]
