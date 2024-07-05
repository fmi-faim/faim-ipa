import pytest

from faim_ipa.mobie import Project


@pytest.fixture
def project(tmp_path):
    return Project.create(
        parent=tmp_path,
        name="test_fixture_project",
        description="A dummy project used for code testing",
    )


def test_create_project(tmp_path):
    project = Project.create(
        parent=tmp_path, name="test_project", description="Dummy description"
    )
    assert (project.path / "project.json").exists()


def test_create_dataset(project):
    dataset = project.create_dataset(
        name="dataset1",
        description="The first test dataset",
        is2d=True,
    )
    assert (dataset.path / "dataset.json").exists()
