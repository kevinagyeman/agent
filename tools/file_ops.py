from typing import Dict, Any
from pathlib import Path

async def read_file(working_directory: Path, path: str) -> Dict[str, Any]:
    try:
        file_path = (working_directory / path).resolve()
        if not str(file_path).startswith(str(working_directory)):
            return {"error": "Access denied: path outside working directory"}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"success": True, "content": content, "path": str(file_path)}
    except Exception as e:
        return {"error": f"Could not read file: {str(e)}"}
