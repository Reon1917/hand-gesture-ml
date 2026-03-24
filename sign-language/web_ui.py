from __future__ import annotations

from asl_app.runtime_bootstrap import bootstrap_local_venv

bootstrap_local_venv(__file__, ("uvicorn", "PIL", "fastapi"))

import uvicorn

from asl_app.webapp import create_app, parse_args


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
