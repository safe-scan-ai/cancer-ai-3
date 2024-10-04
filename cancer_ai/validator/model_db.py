import bittensor as bt
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from ..chain_models_store import ChainMinerModel

Base = declarative_base()

RETRIEVE_MODELS_TIME_FRAME = 30 # minutes
STORED_MODELS_PER_HOTKEY = 10

class ModelInfoTable(Base):
    __tablename__ = 'models'
    competition_id = Column(String, nullable=False)
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

    def get_latest_models(self, hotkeys: list[str]) -> list[ChainMinerModel]:
        # Set the cutoff time to 30 minutes ago
        cutoff_time = datetime.now() - timedelta(minutes=RETRIEVE_MODELS_TIME_FRAME)

        session = self.Session()
        try:
            # Use a correlated subquery to get the latest record for each hotkey that doesn't violate the cutoff
            latest_models = []
            for hotkey in hotkeys:
                model_record = (
                    session.query(ModelInfoTable)
                    .filter(ModelInfoTable.hotkey == hotkey)
                    .filter(ModelInfoTable.date_uploaded < cutoff_time)
                    .order_by(ModelInfoTable.date_uploaded.desc())  # Order by newest first
                    .first()  # Get the first (newest) record that meets the cutoff condition
                )
                if model_record:
                    latest_models.append(
                        ChainMinerModel(
                            competition_id=model_record.competition_id,
                            hf_repo_id=model_record.hf_repo_id,
                            hf_model_filename=model_record.hf_model_filename,
                            hf_repo_type=model_record.hf_repo_type,
                            hf_code_filename=model_record.hf_code_filename,
                            block=model_record.block
                        )
                    )

            return latest_models
        finally:
            session.close()

    def clean_old_records(self, hotkeys: list[str]):
        session = self.Session()
        try:
            for hotkey in hotkeys:
                # Query all records for this hotkey, ordered by date_uploaded in descending order
                records = (
                    session.query(ModelInfoTable)
                    .filter(ModelInfoTable.hotkey == hotkey)
                    .order_by(ModelInfoTable.date_uploaded.desc())
                    .all()
                )

                # If there are more than STORED_MODELS_PER_HOTKEY records, delete the oldest ones
                if len(records) > STORED_MODELS_PER_HOTKEY:
                    records_to_delete = records[STORED_MODELS_PER_HOTKEY:]
                    for record in records_to_delete:
                        session.delete(record)

            # Delete all records for hotkeys not in the given list
            session.query(ModelInfoTable).filter(ModelInfoTable.hotkey.notin_(hotkeys)).delete(synchronize_session=False)
            session.commit()
        
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()