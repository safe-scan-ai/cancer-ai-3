from typing import List
from pydantic import BaseModel

class CompetitionModel(BaseModel):
    competition_id: str
    category: str | None = None
    evaluation_times: List[str]
    dataset_hf_repo: str
    dataset_hf_filename: str
    dataset_hf_repo_type: str


class CompetitionsListModel(BaseModel):
    competitions: List[CompetitionModel]

