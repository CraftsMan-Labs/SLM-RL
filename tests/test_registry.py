import pytest

from slm_rl.games.base import Game
from slm_rl.games.registry import available_games, get_game


def test_all_five_launch_games_registered():
    games = available_games()
    for name in ["mastermind", "connect4", "blackjack", "atari_freeway", "dominion"]:
        assert name in games


def test_get_game_returns_game_subclass():
    cls = get_game("mastermind")
    assert issubclass(cls, Game)
    assert cls.name == "mastermind"


def test_unknown_game_error_lists_available():
    with pytest.raises(KeyError, match="mastermind"):
        get_game("does-not-exist")
