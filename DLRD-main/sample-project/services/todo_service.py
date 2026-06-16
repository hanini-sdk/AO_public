"""Business logic for managing todos (Service layer)."""

from data.models import Todo
from data.repository import TodoRepository
from utils.helpers import slugify


class TodoService:
    """Coordinates todo creation, listing and completion."""

    def __init__(self) -> None:
        self._repo = TodoRepository()

    def add(self, title: str) -> Todo:
        todo = Todo(id=self._repo.next_id(), title=title, slug=slugify(title))
        self._repo.save(todo)
        return todo

    def complete(self, todo_id: int) -> bool:
        todo = self._repo.get(todo_id)
        if todo is None:
            return False
        todo.done = True
        self._repo.save(todo)
        return True

    def list_todos(self) -> list[Todo]:
        return self._repo.all()
