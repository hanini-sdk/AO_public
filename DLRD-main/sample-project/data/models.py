"""Data models for the Todo app (Data layer)."""

from dataclasses import dataclass


@dataclass
class Todo:
    id: int
    title: str
    slug: str
    done: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "slug": self.slug, "done": self.done}
