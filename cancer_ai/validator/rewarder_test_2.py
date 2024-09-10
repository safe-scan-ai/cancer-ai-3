import pytest
from datetime import datetime, timedelta, timezone
from .rewarder import CompetitionLeader, Score, CompetitionWinnersStore, Rewarder
from cancer_ai.validator.competition_handlers.base_handler import ModelEvaluationResult
import numpy as np


@pytest.mark.asyncio
async def test_winner_results_model_improved():
    """
    Set new leader if winner's model has better scores
    """
    current_model_results = ModelEvaluationResult(
        score=0.90,
    )

    new_model_results = ModelEvaluationResult(
        score=0.99,
    )

    competition_leaders = {
        "competition1": CompetitionLeader(
            hotkey="player_1",
            leader_since=datetime.now() - timedelta(days=30 + 3 * 7),
            model_result=current_model_results,
        ),
    }

    scores = {
        "player_1": Score(score=1.0, reduction=0.0),
    }

    winners_store = CompetitionWinnersStore(
        competition_leader_map=competition_leaders, hotkey_score_map=scores
    )

    rewarder = Rewarder(winners_store)
    await rewarder.update_scores(
        winner_hotkey="player_2",
        competition_id="competition1",
        winner_model_result=new_model_results,
    )
    assert (
        winners_store.competition_leader_map["competition1"].model_result
        == new_model_results
    )
    assert winners_store.competition_leader_map["competition1"].hotkey == "player_2"


@pytest.mark.asyncio
async def test_winner_empty_store():
    """
    Test rewards if store is empty
    """
    model_results = ModelEvaluationResult(
        score=0.9,
    )
    competition_leaders = {}
    scores = {}

    winners_store = CompetitionWinnersStore(
        competition_leader_map=competition_leaders, hotkey_score_map=scores
    )
    rewarder = Rewarder(winners_store)
    await rewarder.update_scores(
        winner_hotkey="player_1",
        competition_id="competition1",
        winner_model_result=model_results,
    )
    assert (
        winners_store.competition_leader_map["competition1"].model_result
        == model_results
    )


@pytest.mark.asyncio
async def test_winner_results_model_copying():
    """
    Set new leader if winner's model has better scores
    """
    current_model_results = ModelEvaluationResult(
        score=0.9,
    )

    new_model_results = ModelEvaluationResult(
        score=0.9002,
    )

    competition_leaders = {
        "competition1": CompetitionLeader(
            hotkey="player_1",
            leader_since=datetime.now(timezone.utc) - timedelta(days=30 + 3 * 7),
            model_result=current_model_results,
        ),
    }

    scores = {
        "player_1": Score(score=1.0, reduction=0.0),
    }

    winners_store = CompetitionWinnersStore(
        competition_leader_map=competition_leaders, hotkey_score_map=scores
    )

    rewarder = Rewarder(winners_store)
    await rewarder.update_scores(
        winner_hotkey="player_2",
        competition_id="competition1",
        winner_model_result=new_model_results,
    )
    assert (
        winners_store.competition_leader_map["competition1"].model_result.score
        == current_model_results.score
    )
    assert winners_store.competition_leader_map["competition1"].hotkey == "player_1"


if __name__ == "__main__":
    pytest.main()
