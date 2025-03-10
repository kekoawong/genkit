# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Functionality used by the Genkit veneer to start multiple servers.

The following servers may be started depending upon the host environment:

- Reflection API server.
- Flows server.

The reflection API server is started only in dev mode, which is enabled by the
setting the environment variable `GENKIT_ENV` to `dev`. By default, the
reflection API server binds and listens on (localhost, 3100).  The flows server
is the production servers that exposes flows and actions over HTTP.
"""

import atexit
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ServerSpec:
    """ServerSpec encapsulates the scheme, host and port information for a
    server to bind to and listen on."""

    port: int
    scheme: str = 'http'
    host: str = 'localhost'

    @property
    def url(self) -> str:
        """URL evaluates to the host base URL given the server specs."""
        return f'{self.scheme}://{self.host}:{self.port}'


def create_runtime(
    runtime_dir: str,
    reflection_server_spec: ServerSpec,
    at_exit_fn: Callable[[Path], None] | None = None,
) -> Path:
    """Create a runtime configuration for use with the genkit CLI.
    The runtime information is stored in the form of a timestamped JSON file.
    Note that the file will be cleaned up as soon as the program terminates.
    Args:
        runtime_dir: The directory to store the runtime file in.
        reflection_server_spec: The server specification for the reflection
        server.
        at_exit_fn: A function to call when the runtime file is deleted.
    Returns:
        A path object representing the created runtime metadata file.
    """
    if not os.path.exists(runtime_dir):
        os.makedirs(runtime_dir)

    current_datetime = datetime.now()
    runtime_file_name = f'{current_datetime.isoformat()}.json'
    runtime_file_path = Path(os.path.join(runtime_dir, runtime_file_name))
    metadata = json.dumps(
        {
            'id': f'{os.getpid()}',
            'pid': os.getpid(),
            'reflectionServerUrl': reflection_server_spec.url,
            'timestamp': f'{current_datetime.isoformat()}',
        }
    )
    runtime_file_path.write_text(metadata, encoding='utf-8')

    if at_exit_fn:
        atexit.register(lambda: at_exit_fn(runtime_file_path))
    return runtime_file_path


def is_dev_environment() -> bool:
    """Returns True if the current environment is a development environment.
    Returns:
        True if the current environment is a development environment.
    """
    return os.getenv('GENKIT_ENV') == 'dev'
