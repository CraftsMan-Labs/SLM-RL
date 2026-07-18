import pytest

from slm_rl.games.base import Game
from slm_rl.games.registry import available_games, get_game

KEEPERS = [
    "boxing",
    "space-invaders",
    "freeway",
    "demon-attack",
]


def test_keeper_games_registered():
    games = available_games()
    for name in KEEPERS:
        assert name in games
    assert set(KEEPERS) == set(games)


def test_deleted_games_not_registered():
    games = available_games()
    for name in [
        "minesweeper", "mastermind", "pong", "breakout", "connect4",
        "blackjack", "wordle", "2048", "bowling",
    ]:
        assert name not in games


def test_get_game_returns_game_subclass():
    cls = get_game("boxing")
    assert issubclass(cls, Game)
    assert cls.name == "boxing"


def test_unknown_game_error_lists_available():
    with pytest.raises(KeyError, match="boxing"):
        get_game("does-not-exist")
