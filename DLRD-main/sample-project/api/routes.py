"""HTTP-style routing layer for the Todo app (API layer)."""

from services.todo_service import TodoService
from utils.helpers import format_response


def handle_request(method: str, path: str, service: TodoService) -> str:
    """Dispatch a pseudo HTTP request to the service and format the response."""
    if method == "GET" and path == "/todos":
        return format_response(service.list_todos())
    if method == "POST" and path == "/todos":
        return format_response(service.add("Untitled"))
    return format_response({"error": "not found"})
