// A minimal presentational component for the Todo app (UI layer).
import { useState } from "react";

export function TodoList({ todos }) {
  const [filter, setFilter] = useState("all");
  const visible = todos.filter((t) => filter === "all" || (filter === "done") === t.done);
  return (
    <ul className="todo-list">
      {visible.map((t) => (
        <TodoItem key={t.id} todo={t} />
      ))}
    </ul>
  );
}

export function TodoItem({ todo }) {
  return (
    <li className={todo.done ? "done" : ""}>
      {todo.title}
    </li>
  );
}
