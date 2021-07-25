if __name__ == "__main__":

    # Import after installing requirements, otherwise setup doesn't know versionpy_versioning
    from setuptools import setup, find_packages

    setup(
        name="setup",
        description="setup",
        version="0.1",
        packages=find_packages(),
        include_package_data=True,
        test_suite="nose2.collector.collector",
    )