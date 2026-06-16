"""In-memory persistence for todos (Data layer)."""

from data.models import Todo


class TodoRepository:
    """A trivial in-memory store keyed by todo id."""

    def __init__(self) -> None:
        self._items: dict[int, Todo] = {}
        self._counter = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def save(self, todo: Todo) -> None:
        self._items[todo.id] = todo

    def get(self, todo_id: int) -> Todo | None:
        return self._items.get(todo_id)

    def all(self) -> list[Todo]:
        return list(self._items.values())
