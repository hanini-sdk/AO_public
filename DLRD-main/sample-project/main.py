"""Entry point for the sample Todo application."""

from api.routes import handle_request
from services.todo_service import TodoService


def main() -> None:
    service = TodoService()
    service.add("Write the README")
    service.add("Ship the dashboard")
    print(handle_request("GET", "/todos", service))
    service.complete(1)
    print(handle_request("GET", "/todos", service))


if __name__ == "__main__":
    main()
