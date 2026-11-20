from importlib.metadata import version

SERVICE_NAME = "genui-quality-gate"


def application_version() -> str:
    return version(SERVICE_NAME)
