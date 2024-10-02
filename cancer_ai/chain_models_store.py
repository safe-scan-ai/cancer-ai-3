import functools
from typing import Optional, Type

import bittensor as bt
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .utils.models_storage_utils import run_in_subprocess
from datetime import datetime

Base = declarative_base()

class ChainMinerModel(BaseModel):
    """Uniquely identifies a trained model"""

    competition_id: Optional[str] = Field(description="The competition id")
    hf_repo_id: Optional[str] = Field(description="Hugging Face repository id.")
    hf_model_filename: Optional[str] = Field(description="Hugging Face model filename.")
    hf_repo_type: Optional[str] = Field(
        description="Hugging Face repository type.", default="model"
    )
    hf_code_filename: Optional[str] = Field(
        description="Hugging Face code zip filename."
    )
    block: Optional[int] = Field(
        description="Block on which this model was claimed on the chain."
    )

    class Config:
        arbitrary_types_allowed = True

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.hf_repo_id}:{self.hf_model_filename}:{self.hf_code_filename}:{self.competition_id}:{self.hf_repo_type}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ChainMinerModel"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        if len(tokens) != 5:
            return None
        return cls(
            hf_repo_id=tokens[0],
            hf_model_filename=tokens[1],
            hf_code_filename=tokens[2],
            competition_id=tokens[3],
            hf_repo_type=tokens[4],
            block=None,
        )

class ChainMinerModelStore(BaseModel):
    hotkeys: dict[str, ChainMinerModel | None]
    last_updated: float | None = None

class ChainModelMetadata:
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        netuid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.netuid = netuid

    async def store_model_metadata(self, model_id: ChainMinerModel):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            self.subtensor.commit,
            self.wallet,
            self.netuid,
            model_id.to_compressed_str(),
        )
        run_in_subprocess(partial, 60)

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ChainMinerModel]:
        """Retrieves model metadata on this subnet for specific hotkey"""
        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        try:
            metadata = bt.extrinsics.serving.get_metadata(
                self.subtensor, self.netuid, hotkey
            )
        except Exception as e:
            bt.logging.error(f"Error retrieving metadata for hotkey {hotkey}: {e}")
            return None
        if not metadata:
            return None
        bt.logging.trace(f"Model metadata: {metadata['info']['fields']}")
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        chain_str = bytes.fromhex(hex_data).decode()
        try:
            model = ChainMinerModel.from_compressed_str(chain_str)
            bt.logging.debug(f"Model: {model}")
            if model is None:
                bt.logging.error(
                    f"Metadata might be in old format on the chain for hotkey {hotkey}. Raw value: {chain_str}"
                )
                return None
        except:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.error(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}. Raw value: {chain_str}"
            )
            return None
        # The block id at which the metadata is stored
        model.block = metadata["block"]
        return model

class ModelInfoTable(Base):
    __tablename__ = 'models'
    competition_id = Column(Integer, nullable=False)
    hf_repo_id = Column(String, nullable=False)
    hf_model_filename = Column(String, nullable=False)
    hf_repo_type = Column(String, nullable=False)
    hf_code_filename = Column(String, nullable=False)
    date_uploaded = Column(DateTime, nullable=False)
    block = Column(Integer, nullable=False)
    hotkey = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('date_uploaded', 'hotkey', name='pk_date_hotkey'),
    )

class ModelPersister:
    def __init__(self, subtensor: bt.subtensor, db_url='sqlite:///models.db'):
        self.subtensor = subtensor
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_model(self, chain_miner_model: ChainMinerModel, hotkey: str):
        session = self.Session()
        date_uploaded = self.get_block_timestamp(chain_miner_model.block)
        existing_model = self.get_model(date_uploaded, hotkey)
        if not existing_model:
            try:
                model_record = ModelInfoTable(
                    competition_id = chain_miner_model.competition_id,
                    hf_repo_id = chain_miner_model.hf_repo_id,
                    hf_model_filename = chain_miner_model.hf_model_filename,
                    hf_repo_type = chain_miner_model.hf_repo_type,
                    hf_code_filename = chain_miner_model.hf_code_filename,
                    date_uploaded = date_uploaded,
                    block = chain_miner_model.block,
                    hotkey = hotkey
                )
                session.add(model_record)
                session.commit()
                bt.logging.debug(f"Successfully added model info for hotkey {hotkey} into the DB.")
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        else:
            bt.logging.debug(f"Model for hotkey {hotkey} and date {date_uploaded} already exists, skipping.")
    
    def get_model(self, date_uploaded: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ModelInfoTable).filter_by(
                date_uploaded=date_uploaded, hotkey=hotkey
            ).first()
            if model_record:
                return ChainMinerModel(
                    competition_id = model_record.competition_id,
                    hf_repo_id = model_record.hf_repo_id,
                    hf_model_filename = model_record.hf_model_filename,
                    hf_repo_type = model_record.hf_repo_type,
                    hf_code_filename = model_record.hf_code_filename,
                    block = model_record.block
                )
            return None
        finally:
            session.close()

    def delete_model(self, date_uploaded: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ModelInfoTable).filter_by(
                date_uploaded=date_uploaded, hotkey=hotkey
            ).first()
            if model_record:
                session.delete(model_record)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_block_timestamp(self, block_number):
        """Gets the timestamp of a block given its number."""
        try:
            block_hash = self.subtensor.get_block_hash(block_number)
            if block_hash is None:
                raise ValueError(f"Block hash not found for block number {block_number}")
            
            timestamp_info = self.subtensor.substrate.query(
                module='Timestamp',
                storage_function='Now',
                block_hash=block_hash
            )

            if timestamp_info is None:
                raise ValueError(f"Timestamp not found for block hash {block_hash}")

            timestamp_ms = timestamp_info.value
            block_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0)

            return block_datetime
        except Exception as e:
            bt.logging.error(f"Error retrieving block timestamp: {e}")
            raise
