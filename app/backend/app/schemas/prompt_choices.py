from pydantic import BaseModel, Field, model_validator


class PromptChoicePin(BaseModel):
    choice_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    options_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    option_index: int = Field(ge=0)


class PromptChoicePins(BaseModel):
    prompt_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    items: list[PromptChoicePin] = Field(default_factory=list, max_length=4096)

    @model_validator(mode='after')
    def unique_choices(self):
        if len({item.choice_id for item in self.items}) != len(self.items):
            raise ValueError('prompt_choices_duplicate')
        return self
