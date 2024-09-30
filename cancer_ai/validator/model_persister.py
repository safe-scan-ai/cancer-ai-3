import bittensor as bt
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ModelInfo(BaseModel):
    model_reference: str
    date_uploaded: datetime
    hotkey: str

class ModelTable(Base):
    __tablename__ = 'models'

    model_link = Column(String, nullable=False)
    date_uploaded = Column(DateTime, nullable=False)
    hotkey = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('date_uploaded', 'hotkey', name='pk_date_hotkey'),
    )

class ModelPersister:
    def __init__(self, db_url='sqlite:///models.db'):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_model(self, model_info: ModelInfo):
        session = self.Session()
        try:
            model_record = ModelTable(
                model_reference=model_info.model_reference,
                date_uploaded=model_info.date_uploaded,
                hotkey=model_info.hotkey
            )
            session.add(model_record)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_model(self, date_uploaded: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ModelTable).filter_by(
                date_uploaded=date_uploaded, hotkey=hotkey
            ).first()
            if model_record:
                return {
                    'model_reference': model_record.model_reference,
                    'date_uploaded': model_record.date_uploaded,
                    'hotkey': model_record.hotkey
                }
            return None
        finally:
            session.close()

    def delete_model(self, date_uploaded: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ModelTable).filter_by(
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

    async def get_block_timestamp(self, block_number):
        """Gets the timestamp of a block given its number."""
        block_hash = self.subtensor.get_block_hash(block_number)
        if block_hash is None:
            bt.logging.error(f"Block hash not found for block number {block_number}")
            return None
        
        timestamp_info = self.subtensor.substrate.query(
            module='Timestamp',
            storage_function='Now',
            block_hash=block_hash
        )

        if timestamp_info is None:
            bt.logging.error(f"Timestamp not found for block hash {block_hash}")
            return None

        timestamp_ms = timestamp_info.value
        block_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0)

        return block_datetime