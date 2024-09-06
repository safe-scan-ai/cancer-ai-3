# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import asyncio
import os
import traceback

import bittensor as bt
import numpy as np
import wandb
from cancer_ai.chain_models_store import ChainModelMetadataStore, ChainMinerModelMapping
from cancer_ai.validator.rewarder import WinnersMapping, Rewarder, Score
from cancer_ai.base.base_validator import BaseValidatorNeuron
from cancer_ai.validator.competition_manager import CompetitionManager
from competition_runner import (
    get_competitions_schedule,
    run_competitions_tick,
    CompetitionRunStore,
)


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.competition_scheduler = get_competitions_schedule(
            self.config, self.subtensor, self.hotkeys, test_mode=False
        )
        bt.logging.info(f"Scheduler config: {self.competition_scheduler}")

        self.rewarder = Rewarder(self.winners_mapping)
        

    async def concurrent_forward(self):
        coroutines = [
            self.competition_loop_tick(),
            self.refresh_miners(),
        ]
        await asyncio.gather(*coroutines)

    async def refresh_miners(self):
        """
        Updates hotkeys and downloads information of models from the chain
        """
        bt.logging.info("Synchronizing miners from the chain")
        bt.logging.info(f"Amount of hotkeys: {len(self.hotkeys)}")
        for hotkey in self.hotkeys:
            hotkey_metadata = (
                await self.miners_models.retrieve_model_metadata(hotkey)
            )
            if not hotkey_metadata:
                bt.logging.warning(
                    f"Cannot get miner model for hotkey {hotkey} from the chain, skipping"
                )
                continue
            try:
                chain_miner_model = await self.get_miner_model(hotkey)
                self.chain_miner_models[hotkey] = hotkey_metadata
                self.model_manager.hotkey_store[hotkey] = chain_miner_model
            except ValueError:
                bt.logging.error(
                    f"Miner {hotkey} with data  {hotkey_metadata.to_compressed_str()} does not belong to this competition, skipping"
                )
        bt.logging.info(
            f"Amount of chain miners with models: {len(self.chain_miner_models)}"
        )
        

    async def competition_loop_tick(self):
        # resync the config for scheduler
        self.run_log = CompetitionRunStore(runs=[])

        print("yo")
        self.competition_scheduler = get_competitions_schedule(
            self.config, self.subtensor, self.hotkeys, test_mode=True
        )
        try:
            winning_hotkey, competition_id = await run_competitions_tick(
                self.competition_scheduler, self.run_log
            )
        except Exception:
            formatted_traceback = traceback.format_exc()
            bt.logging.error(f"Error running competition: {formatted_traceback}")
            wandb.init(
                reinit=True, project="competition_id", group="competition_evaluation"
            )
            wandb.log(
                {
                    "winning_evaluation_hotkey": "",
                    "run_time": "",
                    "validator_id": self.wallet.hotkey.ss58_address,
                    "errors": str(formatted_traceback),
                }
            )
            wandb.finish()
            return

        if not winning_hotkey:
            return

        wandb.init(reinit=True, project=competition_id, group="competition_evaluation")
        run_time_s = (
            self.run_log.runs[-1].end_time - self.run_log.runs[-1].start_time
        ).seconds
        wandb.log(
            {
                "winning_hotkey": winning_hotkey,
                "run_time_s": run_time_s,
                "validator_id": self.wallet.hotkey.ss58_address,
                "errors": "",
            }
        )
        wandb.finish()

        bt.logging.info(f"Competition result for {competition_id}: {winning_hotkey}")

        # update the scores
        await self.rewarder.update_scores(winning_hotkey, competition_id)
        self.winners_mapping = WinnersMapping(
            competition_leader_map=self.rewarder.competition_leader_mapping,
            hotkey_score_map=self.rewarder.scores,
        )
        self.save_state()

        hotkey_to_score_map = self.winners_mapping.hotkey_score_map

        self.scores = [
            np.float32(
                hotkey_to_score_map.get(hotkey, Score(score=0.0, reduction=0.0)).score
            )
            for hotkey in self.metagraph.hotkeys
        ]
        self.save_state()

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        if not getattr(self, "winners_mapping", None):
            self.winners_mapping = WinnersMapping(
                competition_leader_map={}, hotkey_score_map={}
            )
        if not getattr(self, "run_log", None):
            self.run_log = CompetitionRunLog(runs=[])
        if not getattr(self, "miners_models", None):
            self.miners_models = ChainMinerModelMapping(hotkeys=[])
        
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            scores=self.scores,
            hotkeys=self.hotkeys,
            rewarder_config=self.winners_mapping.model_dump(),
            run_log=self.run_log.model_dump(),
            miners_models=self.miners_models,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        if not os.path.exists(self.config.neuron.full_path + "/state.npz"):
            bt.logging.info("No state file found. Creating the file.")
            np.savez(
                self.config.neuron.full_path + "/state.npz",
                scores=self.scores,
                hotkeys=self.hotkeys,
                rewarder_config=self.winners_mapping.model_dump(),
                run_log=self.run_log.model_dump(),
            )
            return

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz", allow_pickle=True)
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        self.winners_mapping = WinnersMapping.model_validate(
            state["rewarder_config"].item()
        )
        self.run_log = CompetitionRunLog.model_validate(state["run_log"].item())
        self.miners_models= ChainMinerModelMapping(hotkeys=state["miners_models"])


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
