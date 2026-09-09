from __future__ import annotations

import multiprocessing
import os
import threading
import traceback
from pathlib import Path
from typing import Any

_CELLCHAT_SPAWN_LOCK = threading.Lock()


def _cellchat_child(send_connection: Any, action: str, payload: dict[str, Any]) -> None:
    try:
        r_home = payload.pop("cellchat_r_home", None)
        r_libs_user = payload.pop("cellchat_r_libs_user", None)
        if r_home is not None:
            os.environ["R_HOME"] = str(r_home)
        if r_libs_user is not None:
            os.environ["R_LIBS_USER"] = str(r_libs_user)
        from .cellchat import CellChat

        cellchat = CellChat()
        if action == "database":
            result = cellchat.database_interactions(), cellchat.database_genes()
        elif action == "run":
            result = cellchat.run(payload["adata"], **payload["kwargs"])
        else:
            raise ValueError(f"Unknown CellChat subprocess action: {action}")
        send_connection.send(("ok", result))
    except BaseException:
        send_connection.send(("error", traceback.format_exc()))
    finally:
        send_connection.close()


def _cellchat_isolated(action: str, payload: dict[str, Any]) -> Any:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_cellchat_child,
        args=(send_connection, action, payload),
    )
    r_home = payload.get("cellchat_r_home")
    r_libs_user = payload.get("cellchat_r_libs_user")
    spawn_environment = {
        "R_HOME": None if r_home is None else str(r_home),
        "R_LIBS_USER": None if r_libs_user is None else str(r_libs_user),
    }
    if r_home is not None:
        r_library_path = str(Path(r_home) / "lib")
        current_library_path = os.environ.get("LD_LIBRARY_PATH")
        spawn_environment["LD_LIBRARY_PATH"] = (
            r_library_path
            if not current_library_path
            else f"{r_library_path}:{current_library_path}"
        )
    with _CELLCHAT_SPAWN_LOCK:
        previous_environment = {
            name: os.environ.get(name) for name in spawn_environment
        }
        try:
            for name, value in spawn_environment.items():
                if value is not None:
                    os.environ[name] = value
            process.start()
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    send_connection.close()
    try:
        if not receive_connection.poll(7200):
            raise TimeoutError(f"CellChat subprocess timed out during {action}")
        try:
            status, result = receive_connection.recv()
        except EOFError as error:
            raise RuntimeError(f"CellChat subprocess exited during {action}") from error
    finally:
        receive_connection.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    if status != "ok":
        raise RuntimeError(f"CellChat subprocess failed during {action}:\n{result}")
    return result
